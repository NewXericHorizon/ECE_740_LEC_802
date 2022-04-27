from __future__ import absolute_import

import os
from got10k.experiments import ExperimentOTB, ExperimentGOT10k, ExperimentVOT
from mindspore import context
from tracker import SiamFCTracker


if __name__ == '__main__':
    net_path = 'models/siamfc/SiamFC-50_1166.ckpt'
    context.set_context(
        mode=context.GRAPH_MODE,
        device_id=0,
        save_graphs=False,
        device_target='Ascend')
    tracker = SiamFCTracker(model_path=net_path)
    
    root_dir = os.path.expanduser('VOT2018')
    e = ExperimentVOT(root_dir, version=2018)
    
#     root_dir = os.path.expanduser('OTB-100')
#     e = ExperimentOTB(root_dir, version=2013)
    
#     root_dir = os.path.expanduser('dataset')
#     e = ExperimentGOT10k(root_dir, subset='test')
    
    e.run(tracker)
    e.report([tracker.name])