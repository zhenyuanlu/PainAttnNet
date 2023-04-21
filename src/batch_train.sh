# For windows: put jq-win64.exe in parent folder if other methods don't work
alias jq='../jq/jq-win64.exe'

# For conda: set the path to the anaconda environment
PATH=$PATH:/d/CreativeCenter/anaconda3/envs/PAN/

start=0
end=`cat config.json | jq '.data_loader.args.num_folds'`
end=$((end-1))

# Loop over all the folds and trainers the model
for i in $(eval echo {$start..$end})
do
   python train_kfold_cv.py --fold_id=$i
done
echo '========= Model has been trained ========='