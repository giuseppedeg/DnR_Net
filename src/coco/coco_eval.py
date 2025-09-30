import os
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


PRED_FILE = "out_multi/001.json"  
PRED_FILE = "out/ann.json"  


GT_FILE = "data/icdar_test/data.json"


def summarizeCustom(self, verbose=True):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100, verbose=True):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        if verbose:
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    
    def _summarizeCustom(verbose):
        stats = np.zeros((1,))
        stats[0] = _summarize(1, maxDets=self.params.maxDets[0], verbose=verbose)
        return stats


    self.stats = _summarizeCustom(verbose=verbose)

COCOeval.summarizeCustom = summarizeCustom


    

    
if __name__ == '__main__':
    jFile = open(os.path.join(PRED_FILE))
    predictions = json.load(jFile)
    jFile.close()

    jFile = open(os.path.join(GT_FILE))
    gt = json.load(jFile)
    jFile.close()

    imgid_coding = {}
    imgid_ending_pred = {}

    for img in predictions['images']:
        imgid_ending_pred[img['id']] = os.path.basename(img['file_name'])

    for img in gt['images']:
        if os.path.basename(img['file_name']) in imgid_ending_pred.values():
            imgid_coding[os.path.basename(img['file_name'])] = img['id']
        
   
    for annotation in list(gt['annotations']) :
        if annotation['image_id'] not in imgid_coding.values() or ("tags" in annotation and annotation['tags']['BaseType'][0] != 'bt1' and annotation['tags']['BaseType'][0] != 'bt2'):
            gt['annotations'].remove(annotation)
        

    for annotation in gt['annotations']:
        annotation['iscrowd'] = 0

    


    for annotation in predictions['annotations']:
        if 'score' not in annotation:
            annotation['score'] = 1
        annotation['image_id'] = imgid_coding[imgid_ending_pred[annotation['image_id']]]
        
    # for annotation in list(predictions['annotations']) :
    #     if annotation['tags']['BaseType'][0] != 'bt1'  and  annotation['tags']['BaseType'][0] != 'bt2':
    #         predictions['annotations'].remove(annotation)


    with open("gt_tmp.json", "w") as outfile:
        json.dump(gt, outfile, indent=4)

    with open("pr_tmp.json", "w") as outfile:
        json.dump(predictions['annotations'], outfile, indent=4)


    cocoGt = COCO('gt_tmp.json')
    cocoDt = cocoGt.loadRes("pr_tmp.json") # pass only the annotations

    os.remove('gt_tmp.json')
    os.remove('pr_tmp.json')


    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.maxDets = [10000]
    

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarizeCustom()
    labelscore = cocoEval.stats[0]


    cocoEval.params.useCats = False

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarizeCustom()
    nolabelscore = cocoEval.stats[0]


    print("Score (no labels) : " + str(nolabelscore))
    print("Score (labels) : " + str(labelscore))

