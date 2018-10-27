# LOCAL
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset"), type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=1, type=int)
    parser.add_argument('-te', '--testing_file', help='Path to testing txt file', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset/directions-2-test.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset/directions-2-train.txt"), type=str)

# REMOTE
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release"), type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=1, type=int)
    parser.add_argument('-te', '--testing_file', help='Path to testing txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/test_videos.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/train_videos.txt"), type=str)

    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-a', '--aside', help='Set percentage of dataset to be used for threshold estimation', required=False, default=False, type=float)
    parser.add_argument('-b', '--bagging', help='Determine whether to run single or bassing-based approach', required=False, default=False, type=int)
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC', type=str)
    parser.add_argument('-d', '--drop_frames', help='Skip some frames for training', required=False, default=False, type=int)
    parser.add_argument('-e', '--error_outcome', help='Json containing output APCER and BPCER', required=False, default='saves/ERROR', type=str)
    parser.add_argument('-i', '--instances', help='Number of samples per bagging model', required=False, default=50, type=int)
    parser.add_argument('-m', '--max_frames', help='Establish maximum number of frames for training', required=False, default=False, type=int)
    parser.add_argument('-s', '--scenario', help='Choose protocol execution', required=False, default='one', type=str)
    parser.add_argument('-p', '--probe_file', help='Path to probe txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-probe.npy"), type=str)
    parser.add_argument('-t', '--train_file', help='Path to train txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/Features/SiW-train.npy"), type=str)
    parser.add_argument('-th', '--threshold', help='Set threshold for probe prediction', required=False, default=0.0, type=float)