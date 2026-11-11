cd /cluster/work/projects/ec12/jinyueg/ambi_ssl/data
wget https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
unzip -q wham_noise.zip
rm wham_noise.zip
echo "Done! WHAM noise data downloaded and unziped at /cluster/work/projects/ec12/jinyueg/ambi_ssl/data/wham_noise"
echo "use 'python3 pack_wham_data.py' to pack the data into shards"
