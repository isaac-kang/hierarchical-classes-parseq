export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4


python test_save_error_analysis.py pretrained=parseq --unsolvable_csv=./google_sheet/label_noise_illegible.csv
