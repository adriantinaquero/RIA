import pyautogui, time
try:
    while True:
        print(pyautogui.position(), end="\r")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass