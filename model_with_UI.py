import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
def get_class(model,data):
    classes =['Autos & Vehicles','Food & Drink','Pets & Animals','Science & Education','Sports']
    model.eval()
    output = model(Google(data))
    #print(output)
    pred = None
    pred = output.max(1, keepdim=True)[1][0][0]
    
    #for predict_batch in predict:
    #    for predicts in predict_batch:
     #       real_predict.append(classes[predicts.item()])
    #print(pred)
    return classes[pred]

def load_picture(path):
    image = Image.open(path)
    
   # data_transform = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor()])
    #validation_set = datasets.ImageFolder(path, transform=data_transform)
    #transform = transforms.RandomResizedCrop(224)
    #image = transform(image)
    validation_set=transforms.functional.to_tensor(image)
    validation_set=validation_set.unsqueeze(0)
    return validation_set


class ANNClassifier_GOOGLE(nn.Module):
    def __init__(self):
        super(ANNClassifier_GOOGLE, self).__init__()
        self.fc1 = nn.Sequential(nn.Dropout(0.5),nn.Linear(1000,6))
        #self.fc2 = nn.Linear(10, 6)
    def forward(self, x):
        x = x.view(-1, 1000) #flatten feature data
        x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return x


class ChildWindow(QDialog):
    def __init__(self,text):
        super().__init__()
        self.initUI(text)
 
    def initUI(self,text):
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        font=QFont()
        font.setPointSize(20)
        self.setWindowTitle("prediction")
        self.label = QLabel(self)
        self.label.setText(text)
        self.label.setFont(font)
        self.label.setParent(self)
        

        self.resize(280, 230)

class PhotoLabel(QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
        QLabel {
            border: 4px dashed #aaa;
        }''')

    def setPixmap(self, *args, **kwargs):
        super().setPixmap(*args, **kwargs)
        self.setStyleSheet('''
        QLabel {
            border: none;
        }''')
class Template(QWidget):

    def __init__(self):
        super().__init__()
        self.photo = PhotoLabel()
        self.finished=False
        btn = QPushButton('Prediction',self)
        btn.clicked.connect(self.btnClicked)
        grid = QGridLayout(self)
        self.text=None
        grid.addWidget(btn, 0, 0, Qt.AlignHCenter)
        grid.addWidget(self.photo, 1, 0)
        self.setAcceptDrops(True)
        self.resize(300, 200)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()
    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            filename = event.mimeData().urls()[0].toLocalFile()
            event.accept()
            self.open_image(filename)
        else:
            event.ignore()

    def open_image(self, filename=None):
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Select Photo', QDir.currentPath(), 'Images (*.png *.jpg)')
            if not filename:
                return
        self.photo.setPixmap(QPixmap(filename))
        print(filename)
        self.text = get_class(model,load_picture(filename))
        self.finished = True
    def btnClicked(self):
        self.chile_Win = ChildWindow(self.text)
        self.chile_Win.show()
        self.chile_Win.resize(500,500)
        self.chile_Win.exec_

if __name__ == '__main__':
    googleNet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
    Google = googleNet
    model = None
    model = ANNClassifier_GOOGLE()
    model.load_state_dict(torch.load(r'C:\Users\tongt\Desktop\model24',map_location=torch.device('cpu')))
    model.eval()
    torch.manual_seed(0)
    app = QApplication(sys.argv)
    gui = Template()
    gui.show()
    gui.setGeometry(20,30,1920,1080)
    sys.exit(app.exec_())

    #Cite: https://stackoverflow.com/questions/60614561/how-to-ask-user-to-input-an-image-in-pyqt5
