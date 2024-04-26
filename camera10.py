import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import torch
from spikingjelly.activation_based import  surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
import torchvision
import matplotlib.pyplot as plt

# MARK: Open the camera, just use first detected DAVIS camera
camera = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS) 

# Initialize a multi-stream slicer
slicer = dv.EventMultiStreamSlicer("events")

# Add a frame stream to the slicer
slicer.addFrameStream("frames")

# Initialize a visualizer for the overlay
visualizer = dv.visualization.EventVisualizer(camera.getEventResolution(), dv.visualization.colors.white(),
                                              dv.visualization.colors.green(), dv.visualization.colors.red())

# Create a figure for trend plot
plt.figure()

# 获取当前图形的管理器
manager = plt.get_current_fig_manager()

# 计算窗口的新位置
window_width = 890
window_height = 653
window_x = 700
window_y = 70

# 设置窗口的位置和大小
manager.window.resize(window_width, window_height)
manager.window.move(window_x, window_y)

# Initialize lists to store data for trend plot
time_data = []
prediction_data = []
filter_prediction_data = []
filter_prediction = 0
legend = False

x_item = ["Hand Clapping", "Right Hand Wave", "Left Hand Wave", "Right Arm\n Clockwise", "Right Arm\n Counter Clockwise", "Left Arm\n Clockwise", "Left Arm \nCounter Clockwise", "Forearm Roll\n Forward", "Forearm Roll \nBackward", "Drums", "Air Guitar", "Other Gestures"]

# Create a window for image display
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# MARK: 创建摄像头对象
camera_real = cv.VideoCapture(0)  # 参数为摄像头索引，0表示第一个摄像头
# 检查摄像头是否成功打开
if not camera_real.isOpened():
    print("无法打开摄像头")
    exit()

def open_real_cam():
    ret, frame = camera_real.read()  # 读取图像帧
    if not ret:
        print("无法获取摄像头的图像")

    cv.namedWindow("Camera", cv.WINDOW_NORMAL)
    cv.imshow("Camera", frame)  # 显示图像

    # 按下 'q' 键退出循环
    if cv.waitKey(1) & 0xFF == ord('q'):
        exit(0)
        
# MARK: 加载模型的权重
model_weights = 'G:\PolyU\CapstoneProject\Project_New\spikingjelly\logs\T16_b16_adam_lr0.001_c128_amp_cupy\checkpoint_max.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
net = parametric_lif_net.DVSGestureNet(channels=128, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
try:
    net.load_state_dict(torch.load(model_weights, map_location=device)['net'])
    net.to(device)
    net.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model weights file not found.")
    exit()
except Exception as e:
    print("Error loading model:", str(e))
    exit()

cnt = 0
#  MARK: Process event 
# Callback method for time based slicing
def process_events(events):
    #extracted_data = ((event.x(), event.y(), 1 if event.polarity() else 0) for event in itertools.islice(events, 1000))
    extracted_data = tuple((event.x(), event.y(), 1 if event.polarity() else 0) for event in events)
    tensor_data = torch.zeros((2, 260, 346), device=device)  # [2, 260, 346]
    #for y, x, p in extracted_data:
    for y, x, p in extracted_data:
        if 0 <= x < 260 and 0 <= y < 346:
            tensor_data[p, x, y] += 1.  
        else:
            print(f"x: {x} y: {y} p: {p}")
    resize = torchvision.transforms.Resize((128, 128), antialias=True) 
    tensor_data = resize(tensor_data).unsqueeze(0).unsqueeze(0)   # [2, 260, 346] -> [1, 1, 2, 128, 128]
    return tensor_data

# MARK: Process frame
def process_frame(frames):
    if len(frames) > 0:
        latest_image = frames[-1].image
        if len(latest_image.shape) == 2:
            latest_image = cv.cvtColor(latest_image, cv.COLOR_GRAY2BGR)
        return latest_image
    return None
# MARK: Display preview Prediction
def display_preview(data):
    global cnt, filter_prediction, legend
    events = data.getEvents("events")
    tensor_data = process_events(events)
    with torch.no_grad():
        tensor_data = tensor_data.to(device)
        tensor_data = tensor_data.squeeze(1)
        output = net(tensor_data).mean(0)
        predicted_label = output.argmax().item()
        if predicted_label != 0:
            if predicted_label == 1:
                predict_output = f"Hand Clapping"
            elif predicted_label == 2:
                predict_output = f"Right Hand Wave"
            elif predicted_label == 3:
                predict_output = f"Left Hand Wave"
            elif predicted_label == 4:
                predict_output = f"Right Arm Clockwise"
            elif predicted_label == 5:
                predict_output = f"Right Arm Counter Clockwise"
            elif predicted_label == 6:
                predict_output = f"Left Arm Clockwise"
            elif predicted_label == 7:
                predict_output = f"Left Arm Counter Clockwise"
            elif predicted_label == 8:
                predict_output = f"Forearm Roll Forward"
            elif predicted_label == 9:
                predict_output = f"Forearm Roll Backward"
            elif predicted_label == 10:
                predict_output = f"Drums"
            elif predicted_label == 11:
                predict_output = f"Air Guitar"
            elif predicted_label == 12:
                predict_output = f"Other Gestures"
            print("predicted gesture: ", predict_output, predicted_label)
            if len(filter_prediction_data) <= 10:
                filter_prediction_data.append(predicted_label)
            else:
                filter_prediction = max(set(filter_prediction_data), key = filter_prediction_data.count)
                filter_prediction_data.pop(0)  # Remove the oldest element 
               
            
    
    frames = data.getFrames("frames")
    latest_image = process_frame(frames)
    if latest_image is None:
        return
    
    if filter_prediction != 0:  
        if filter_prediction == 1:
            label = f"Hand Clapping"
        elif filter_prediction == 2:
            label = f"Right Hand Wave"
        elif filter_prediction == 3:
            label = f"Left Hand Wave"
        elif filter_prediction == 4:
            label = f"Right Arm Clockwise"
        elif filter_prediction == 5:
            label = f"Right Arm Counter Clockwise"
        elif filter_prediction == 6:
            label = f"Left Arm Clockwise"
        elif filter_prediction == 7:
            label = f"Left Arm Counter Clockwise"
        elif filter_prediction == 8:
            label = f"Forearm Roll Forward"
        elif filter_prediction == 9:
            label = f"Forearm Roll Backward"
        elif filter_prediction == 10:
            label = f"Drums"
        elif filter_prediction == 11:
            label = f"Air Guitar"
        elif filter_prediction == 12:
            label = f"Other Gestures"
    else:
        label = f"Predicting..."
        
    label_position = (latest_image.shape[1] - 230, 230) # Position for label
    cv.putText(latest_image, label, label_position, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("Preview", visualizer.generateImage(events, latest_image))
        
    # Append data for trend plot
    time_data.append(cnt)
    prediction_data.append(predicted_label-1)
    # Update and redraw trend plot
    plt.clf()   # Clear the current figure
    #plt.yticks(ticks=range(len(x_item)), labels=x_item)
    plt.plot(time_data, prediction_data, color='blue')  
    plt.xlabel('Time')
    plt.ylabel('Prediction')
    plt.yticks(ticks=range(len(x_item)), labels=x_item)
    plt.title('Gesture Recognition Trend')
    plt.axhline(y=filter_prediction-1, color='red', linestyle='--', label='Predicted Gesture: ' + (label if label != f"Predicting..." else "loading"))    # Draw a horizontal line at the threshold
    plt.legend(loc='upper right', fontsize=18)    # Set the legend position to upper right
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Automatically adjust subplot parameters to fit the figure area
    
    plt.pause(0.001)
    
    if cv.waitKey(2) == 27: # If escape button is pressed (code 27 is escape key), exit the program cleanly
        exit(0)
    cv.waitKey(1)
    
    

# MARK: Register the callback method for time based slicing
slicer.doEveryTimeInterval(timedelta(milliseconds=33), display_preview)   # Display the preview every 33ms

while camera.isRunning():
    events = camera.getNextEventBatch()
    if events is not None:
        slicer.accept("events", events)

    frame = camera.getNextFrame()
    if frame is not None:
        slicer.accept("frames", [frame])
    
    # Increment the counter
    cnt += 1
    
    open_real_cam()

# Show the final trend plot
plt.show()
        