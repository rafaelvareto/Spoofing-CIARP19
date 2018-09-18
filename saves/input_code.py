# LOCAL
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset"), type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=10, type=int)
    parser.add_argument('-te', '--testing_file', help='Path to testing txt file', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset/directions-2-test.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "GIT/Spoofing-VisualRhythm/datasets/SiW-dataset/directions-2-train.txt"), type=str)

# REMOTE
    # Handle arguments
    parser = argparse.ArgumentParser(description='Demo file for running Face Spoofing Detection')
    parser.add_argument('-c', '--chart_path', help='Path to save chart file', required=False, default='saves/ROC_curve.pdf', type=str)
    parser.add_argument('-f', '--folder_path', help='Path to video folder', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release"), type=str)
    parser.add_argument('-e', '--error_outcome', help='Json', required=False, default='saves/error_rates', type=str)
    parser.add_argument('-r', '--repetitions', help='Number of executions [10..INF]', required=False, default=10, type=int)
    parser.add_argument('-te', '--testing_file', help='Path to testing txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/test_videos.txt"), type=str)
    parser.add_argument('-tr', '--training_file', help='Path to training txt file', required=False, default=os.path.join(HOME, "REMOTE/VMAIS/dataset/SiW_release/train_videos.txt"), type=str)