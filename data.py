from function import *
from time import sleep

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    for action in actions:
        print(action)
        for sequence in range(no_sequences):
            print(sequence)
            for frame_num in range(1,sequence_length):
                frame=cv2.imread(f'asl_alphabet_train/{action}/{action} ({frame_num}).jpg', 1)
                print(frame)
                image, results = mediapipe_detection(frame, hands)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)