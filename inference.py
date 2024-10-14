import torch
from mmocr.apis import MMOCRInferencer
from position import Position

# Initialize MMOCR model
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

detModel = './models/detection/dbnet_resnet18_fpnc_1200e_icdar2015.py'
detWeights = './models/detection/dbnet_resnet18_fpnc_1200e_icdar2015_20220825_221614-7c0e94f2.pth'

recModel = './models/recognition/svtr-base_20e_st_mj.py'
recWeights = './models/recognition/svtr-base_20e_st_mj-ea500101.pth'

# Create the inferencer once, so we don't reload it every time a request is made
inferencer = MMOCRInferencer(det=detModel, det_weights=detWeights,
                             rec=recModel, rec_weights=recWeights, 
                             device=DEVICE)

def process_image(image_path):
    """
    Process the image and return the recognized text.
    :param image_path: Path to the image file.
    :return: Recognized text from the image.
    """
    
     # Your inference output
    results = inferencer(image_path)['predictions'][0]

    # Now, we need to adjust the structure to match what the Position class expects:
    # Specifically, pass 'rec_texts', 'rec_scores', and 'det_polygons' directly.
    position = Position()

    # Call the process_predictions method
    final_text = position.process_predictions(results)
        
    return final_text
    
result = process_image('./images/2 lines.jpg')
print(result)