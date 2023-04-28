import single_img
import glob


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

listen_path = './src_imgs'
out_path = './out_imgs'

def is_file_writeable(path):
    try:
        f = open(path, 'a')
        f.close()
    except IOError:
        return False
    return True

class MyHandler(FileSystemEventHandler):
    #def on_any_event(self, event):
    #    print(event.event_type, event.src_path)

    def on_created(self, event):
        #print("on_created", event.src_path)
        #print(event.src_path.strip())
        path = event.src_path.strip()
        # 判断path 是否可写
        while is_file_writeable(path) == False:
            time.sleep(0.1)
            print("wait for writeable")
        try:
            if ".jpg" in path or "jpeg" in path or "png" in path or "jfif" in path:
                single_img.split_single_img(path,out_path)
        except :
            print("error comming")
       
            
       

event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path=listen_path, recursive=True)
observer.start()

input("press any key to exit")

