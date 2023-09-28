from pathlib import Path


def prepare_gt(seq_folder):
    valid_label = [1, 7]
    gt_file = Path(seq_folder)/"gt/gt.txt"
    ngt_file = Path(seq_folder)/"gt/ngt.txt"
    ngt = ngt_file.open("a+")
    with gt_file.open("r") as gt:
        lines = gt.readlines()
        for line in lines:
            f, idx, x, y, w, h, flag, label, vis = line.split(",")
            print(f, idx, x, y, w, h, flag, label, vis)
            if int(label) in valid_label:
                ngt.write
            if int(f) ==2:
                break
    
if __name__ == '__main__':
    prepare_gt("D:\Tracking\MOT16/train\MOT16-02")