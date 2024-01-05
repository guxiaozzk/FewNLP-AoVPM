cuda=0
out_dir='out'
dataset='t2017'            # Options: 't2015', 't2017', 'masad', 'mvsa-s', 'tumemo'1
train_file='few-shot.tsv'  # Options: 'few-shot1.tsv', 'few-shot2.tsv'
template=3        # Options: 1, 2, 3
case $dataset in
    't2015')
        img_dir='IJCAI2019_data/twitter2015_images'
        prompt_shape_pt='333-0'
        prompt_shape_pvlm='333-3'
        early_stop=40
        ;;
    't2017')
        img_dir='IJCAI2019_data/twitter2017_images'
        prompt_shape_pt='333-0'
        prompt_shape_pvlm='333-3'
        early_stop=40
        ;;
    'masad')
        img_dir='MASAD_imgs'
        prompt_shape_pt='333-0'
        prompt_shape_pvlm='333-3'
        early_stop=25
        ;;
    'mvsa-s')
        img_dir='MVSA-S_data'
        prompt_shape_pt='33-0'
        prompt_shape_pvlm='33-1'
        early_stop=40
        ;;
    'mvsa-m')
        img_dir='MVSA-M_data/MVSA/data'
        prompt_shape_pt='33-0'
        prompt_shape_pvlm='33-3'
        early_stop=40
        ;;
    'tumemo')
        img_dir='TumEmo_data'
        prompt_shape_pt='33-0'
        prompt_shape_pvlm='33-3'
        early_stop=15
        ;;
esac

# PVLM5
for img_token_len in 3 3 4 5
do
    for lr in 4e-5 5e-5 3e-5 4e-5 5e-5
    do
        for seed in 5 13 5 17
        do
            python main.py \
                --cuda $cuda \
                --out_dir $out_dir \
                --dataset $dataset \
                --img_dir $img_dir \
                --template $template \
                --prompt_shape $prompt_shape_pvlm \
                --few_shot_file $train_file \
                --img_token_len $img_token_len \
                --batch_size 32 \
                --lr_lm_model $lr \
                --lr_visual_encoder 0 \
                --early_stop $early_stop \
                --seed $seed
        done
    done
done
