import torch
model = torch.hub.load('.', 'custom', path='/Users/railiavaliullina/Documents/GitHub/insulators_absence_detection/'
                                           'training_results/train/weights/best.pt', source='local')
