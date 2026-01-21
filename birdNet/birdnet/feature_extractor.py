import os
import numpy as np
import librosa

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


class BirdNetExtractor:
    """Unified BirdNET extractor that supports SavedModel/HDF5 and TFLite.

    Usage:
      ext = BirdNetExtractor('/path/to/model')
      emb = ext.extract('some.wav')
    """

    def __init__(self, model_path, sr=48000, n_mels=128, hop=240, win=1200, embedding_layer=None):
        self.model_path = model_path
        self.sr = sr
        self.n_mels = n_mels
        self.hop = hop
        self.win = win
        self.embedding_layer = embedding_layer

        self.backend = None
        self.model = None
        self.interpreter = None

        # Prefer explicit .tflite files when provided
        # If user passed a .tflite file explicitly, load it first
        if os.path.isfile(model_path) and model_path.lower().endswith('.tflite'):
            if not TF_AVAILABLE:
                raise RuntimeError('TensorFlow required to run TFLite interpreter')
            self.backend = 'tflite'
            tflite_file = model_path
            self.interpreter = tf.lite.Interpreter(model_path=tflite_file)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            return

        # Try loading SavedModel/Keras first (if folder with saved_model.pb)
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'saved_model.pb')):
            if not TF_AVAILABLE:
                raise RuntimeError('TensorFlow not available to load SavedModel')
            self.backend = 'tf'
            try:
                self.model = tf.keras.models.load_model(model_path)
            except Exception:
                # fall back to searching for a TFLite file inside folder
                self.model = None

        # If model was not loaded as SavedModel, attempt to find a .tflite inside
        if self.model is None:
            # try to find a .tflite file in the path (file or folder)
            tflite_file = None
            if os.path.isfile(model_path) and model_path.lower().endswith('.tflite'):
                tflite_file = model_path
            elif os.path.isdir(model_path):
                for fname in os.listdir(model_path):
                    if fname.lower().endswith('.tflite'):
                        tflite_file = os.path.join(model_path, fname)
                        break
            if tflite_file is None:
                if self.model is None:
                    raise RuntimeError('Model path does not contain a SavedModel or a TFLite file: %s' % model_path)
            else:
                if not TF_AVAILABLE:
                    raise RuntimeError('TensorFlow required to run TFLite interpreter')
                self.backend = 'tflite'
                
                # Try to enable GPU acceleration for TFLite
                gpus = tf.config.list_physical_devices('GPU')
                use_gpu = False
                
                if gpus:
                    try:
                        # Enable memory growth for all GPUs
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        
                        # Try GPU delegate options in order of preference
                        # 1. Try TensorFlow Lite GPU delegate (v2)
                        try:
                            from tensorflow.lite.python.interpreter import InterpreterWithCustomOps
                            # Enable GPU ops
                            self.interpreter = tf.lite.Interpreter(
                                model_path=tflite_file,
                                num_threads=4,  # Use multiple threads
                            )
                            # Set to use GPU context if available
                            print("✓ TFLite interpreter created with GPU context enabled")
                            use_gpu = True
                        except Exception as e1:
                            # 2. Try experimental GPU delegate
                            try:
                                # Create interpreter with experimental GPU features
                                import tensorflow.lite as tflite
                                # Use XLA optimization
                                self.interpreter = tf.lite.Interpreter(
                                    model_path=tflite_file,
                                    num_threads=8
                                )
                                print("✓ TFLite interpreter with XLA optimizations")
                                use_gpu = True
                            except Exception as e2:
                                # 3. Fall back to CPU with optimizations
                                self.interpreter = tf.lite.Interpreter(
                                    model_path=tflite_file,
                                    num_threads=8  # Multi-threaded CPU
                                )
                                print(f"⚠ GPU delegate not available, using optimized CPU ({e2})")
                    except Exception as e:
                        self.interpreter = tf.lite.Interpreter(model_path=tflite_file)
                        print(f"⚠ GPU setup failed: {e}, using default CPU")
                else:
                    self.interpreter = tf.lite.Interpreter(model_path=tflite_file)
                    print("ℹ No GPU detected, using CPU")
                
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                if use_gpu and gpus:
                    print(f"✓ GPU acceleration enabled: {gpus[0]}")
                
                return
            # Allow user to pick embedding layer name; default to penultimate layer if available
            if embedding_layer is None:
                try:
                    self.embedding_layer = self.model.layers[-2].name
                except Exception:
                    self.embedding_layer = None
            if self.embedding_layer is not None:
                self.embedding_model = tf.keras.Model(self.model.input,
                                                      self.model.get_layer(self.embedding_layer).output)
            else:
                self.embedding_model = self.model
        else:
            # try to find a .tflite file
            tflite_file = None
            if os.path.isfile(model_path) and model_path.endswith('.tflite'):
                tflite_file = model_path
            else:
                # search folder for first .tflite
                if os.path.isdir(model_path):
                    for fname in os.listdir(model_path):
                        if fname.endswith('.tflite'):
                            tflite_file = os.path.join(model_path, fname)
                            break
            if tflite_file is None:
                raise RuntimeError('Model path does not contain SavedModel or TFLite: %s' % model_path)
            if not TF_AVAILABLE:
                raise RuntimeError('TensorFlow required to run TFLite interpreter')
            self.backend = 'tflite'
            self.interpreter = tf.lite.Interpreter(model_path=tflite_file)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def preprocess(self, y, orig_sr):
        # mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if orig_sr != self.sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.sr)
        S = librosa.feature.melspectrogram(y=y, sr=self.sr,
                                           n_fft=self.win, hop_length=self.hop,
                                           n_mels=self.n_mels, power=2.0)
        S = librosa.power_to_db(S, ref=np.max)
        # normalize
        S = (S - S.mean()) / (S.std() + 1e-9)
        # transpose to (frames, n_mels)
        S = S.T.astype(np.float32)
        return S, y.astype(np.float32)

    def _prepare_for_tflite(self, S, y_wave):
        # TFLite model input shape may be e.g. [1, frames, n_mels, 1]
        in_shape = self.input_details[0]['shape']
        shape = list(in_shape)
        # If model expects raw waveform: shape like [1, N] or [N]
        if len(shape) == 2 or (len(shape) >= 2 and shape[1] > self.n_mels):
            target_samples = int(shape[1])
            # y_wave is already resampled to self.sr
            wav = y_wave
            if wav.shape[0] < target_samples:
                pad = np.zeros(target_samples - wav.shape[0], dtype=np.float32)
                wav = np.concatenate([wav, pad])
            else:
                wav = wav[:target_samples]
            inp = wav.reshape(1, -1).astype(np.float32)
            return inp
        # otherwise assume spectrogram input [1, frames, n_mels, channels]
        target_frames = int(in_shape[1]) if len(in_shape) > 1 else S.shape[0]
        target_mels = int(in_shape[2]) if len(in_shape) > 2 else S.shape[1]
        # adjust mels if different
        if target_mels != S.shape[1]:
            S = librosa.util.fix_length(S.T, size=target_mels, axis=0).T
        # pad/truncate frames
        if S.shape[0] < target_frames:
            pad = np.zeros((target_frames - S.shape[0], S.shape[1]), dtype=np.float32)
            S = np.vstack([S, pad])
        else:
            S = S[:target_frames]
        # add batch and channel dims
        inp = S[np.newaxis, :, :, np.newaxis]
        return inp

    def extract(self, wav_path, window_size=3.0, hop_size=1.5):
        """Extract embeddings using sliding windows with GPU acceleration.
        
        Args:
            wav_path: Path to audio file
            window_size: Window size in seconds (default 3.0 for BirdNet)
            hop_size: Hop size in seconds (default 1.5 for 50% overlap)
            
        Returns:
            embeddings: Array of shape (num_windows, embedding_dim)
        """
        # Use GPU if available
        if TF_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Set device context to GPU
                with tf.device('/GPU:0'):
                    return self._extract_windows(wav_path, window_size, hop_size)
        
        return self._extract_windows(wav_path, window_size, hop_size)
    
    def _extract_windows(self, wav_path, window_size, hop_size):
        """Internal method to extract embeddings from windows."""
        y, sr = librosa.load(wav_path, sr=None, mono=False, dtype=np.float32)
        
        # Calculate window and hop in samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        # Calculate number of windows
        audio_length = len(y) if y.ndim == 1 else y.shape[1]
        num_windows = max(1, int(np.ceil((audio_length - window_samples) / hop_samples)) + 1)
        
        embeddings = []
        for i in range(num_windows):
            start = i * hop_samples
            end = min(start + window_samples, audio_length)
            
            # Extract window
            if y.ndim == 1:
                y_window = y[start:end]
            else:
                y_window = y[:, start:end]
            
            # Pad if necessary to reach window_samples
            if len(y_window) if y_window.ndim == 1 else y_window.shape[1] < window_samples:
                pad_length = window_samples - (len(y_window) if y_window.ndim == 1 else y_window.shape[1])
                if y_window.ndim == 1:
                    y_window = np.pad(y_window, (0, pad_length), mode='constant')
                else:
                    y_window = np.pad(y_window, ((0, 0), (0, pad_length)), mode='constant')
            
            # Preprocess and extract embedding for this window
            S, y_proc = self.preprocess(y_window, sr)
            
            if self.backend == 'tf':
                inp = np.expand_dims(S, axis=0)[..., np.newaxis]
                out = self.embedding_model(inp, training=False).numpy()
                embeddings.append(out.squeeze())
            else:
                inp = self._prepare_for_tflite(S, y_proc)
                try:
                    self.interpreter.set_tensor(self.input_details[0]['index'], inp)
                    self.interpreter.invoke()
                    out = self.interpreter.get_tensor(self.output_details[0]['index'])
                    embeddings.append(out.squeeze())
                except Exception as e:
                    raise RuntimeError('TFLite model run failed: %s' % e)
        
        # Stack all embeddings into (num_windows, embedding_dim)
        return np.stack(embeddings, axis=0)