#echo "Train the baseline system"
#for seed in 1234 2345 3456 4567 5678; do
#    python3 train.py seed=${seed} exp_name="baseline2024-seed${seed}"
#done


echo "Compare different features"
for seed in 1234; do
    for fea in logmel pcen mel mfcc delta_mfcc pcen@mfcc logmel@mfcc logmel@delta_mfcc pcen@delta_mfcc; do
        for con in true false; do
#   python3 train.py train_param.negative_train_contrast=false seed=${seed} exp_name="baseline2024-no_neg_contrast-seed${seed}"
         python3 train.py seed=${seed} features.feature_types=${fea} train_param.negative_train_contrast=${con} exp_name=JSR${fea}-seed${seed}-NTC${con} 
        done
    done
done


#    python3 train.py features.feature_types="pcen@mfcc" exp_name="pcen@mfcc"
#   python3 train.py features.feature_types="mel" exp_name="mel"
#    python3 train.py features.feature_types="logmel@mfcc" exp_name="logmel@mfcc"
#    python3 train.py features.feature_types="logmel@delta_mfcc" exp_name="logmel@delta_mfcc"
#    python3 train.py features.feature_types="pcen@delta_mfcc" exp_name="pcen@delta_mfcc"



#echo "Train the baseline system w/o negative contrastive"
#for seed in 1234 2345 3456 4567 5678; do
#    python3 train.py train_param.negative_train_contrast=false seed=${seed} exp_name="baseline2024-no_neg_contrast-seed${seed}"
#done