from pathlib import Path


def prepare_gt(seq_folder):
    cnt =0
    valid_label = [1,7]
    gt_file = Path(seq_folder)/"gt/gt.txt"
    ngt_file = Path(seq_folder)/"gt/ngt.txt"
    ngt = ngt_file.open("a+")
    with gt_file.open("r") as gt:
        lines = gt.readlines()
        for line in lines:
            f, idx, x, y, w, h, flag, label, vis = line.split(",")
            if int(label) in valid_label and int(flag) !=0 and float(vis)>=0.5:
                cnt+=1
                ngt.write(f"{f},{idx},{x},{y},{w},{h},-1,-1,-1,-1\n")
    print(cnt)
    ngt.close()
if __name__ == '__main__':
    prepare_gt("D:\Tracking\MOT16/train\MOT16-09")