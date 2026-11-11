from huggingface_hub import snapshot_download

if __name__ == "__main__":
    ## download AmbisonicsRIR
    snapshot_download(
        repo_id="labhamlet/AmbisonicsRIRs",
        local_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/AmbisonicsRIRs",
        repo_type="dataset",
        max_workers=3,
    )

    ## download BinauralRIR sample
    # snapshot_download(
    #     repo_id="labhamlet/BinauralRIRs",
    #     local_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/BinauralRIRs",
    #     repo_type="dataset",
    #     allow_patterns="shard-000000.tar"
    # )

    ## download Natualistic scenes
    # snapshot_download(
    #     repo_id="labhamlet/GRAM-Naturalistic-Scenes-Ambisonics",
    #     local_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/GRAM-Naturalistic-Scenes-Ambisonics",
    #     repo_type="dataset",
    #     max_workers=3,
   #  )

    ## download AudioSet
    # snapshot_download(
    #     repo_id="agkphysics/AudioSet",
    #     local_dir="/cluster/work/projects/ec12/jinyueg/ambi_ssl/data/AudioSet",
    #     repo_type="dataset",
    #     max_workers=3,
    #     revision="5a2fa42a1506470d275a47ff8e1fdac5b364e6ef",
    #     allow_patterns="data/unbal_train**",
    # )
