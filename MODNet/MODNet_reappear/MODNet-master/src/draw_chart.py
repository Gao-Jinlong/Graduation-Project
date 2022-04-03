import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../pretrained/portrait_loss_data.json', 'r') as f:
        illustration_data = json.load(f)
        print(illustration_data)