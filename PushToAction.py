import speech_recognition as sr
import os
import urllib.parse
import wx
import wx.adv

TRAY_TOOLTIP = 'Agador'
TRAY_ICON = 'img/agador.ico'

def create_menu_item(menu, label, func):
    item = wx.MenuItem(menu, -1, label)
    menu.Bind(wx.EVT_MENU, func, id=item.GetId())
    menu.Append(item)
    return item


class TaskBarIcon(wx.adv.TaskBarIcon):
    def __init__(self,frame):
        wx.adv.TaskBarIcon.__init__(self)
        self.myapp_frame = frame
        self.set_icon(TRAY_ICON)
        self.Bind(wx.adv.EVT_TASKBAR_LEFT_DOWN, self.on_left_down)

    def CreatePopupMenu(self):
        menu = wx.Menu()
        create_menu_item(menu, 'Exit', self.on_exit)
        return menu

    def set_icon(self, path):
        icon = wx.Icon(wx.Bitmap(path))
        self.SetIcon(icon, TRAY_TOOLTIP)

    def on_left_down(self, event):
        r = sr.Recognizer()
        mic = sr.Microphone()

        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            os.system("xdg-open https://duckduckgo.com/?q=" + urllib.parse.quote(r.recognize_google(audio)))

    def on_exit(self, event):
        self.myapp_frame.Close()

class My_Application(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, wx.ID_ANY, "", size=(1,1))
        panel = wx.Panel(self)
        self.myapp = TaskBarIcon(self)
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def onClose(self, evt):
        """
        Destroy the taskbar icon and the frame
        """
        self.myapp.RemoveIcon()
        self.myapp.Destroy()
        self.Destroy()

if __name__ == "__main__":
    MyApp = wx.App()
    My_Application()
    MyApp.MainLoop()

