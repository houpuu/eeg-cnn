import tkinter as tk

def callback():
    #global var
    print(var.get())
    var.set("我才不信！")
    print(var.get())

root = tk.Tk()

frame1 = tk.Frame(root)
frame2 = tk.Frame(root)

var = tk.StringVar()
var.set("警告！\n未满18岁,禁止访问！")

textLable = tk.Label(root,
                    textvariable = var,
                    fg="red",
                    padx = 10)
textLable.pack()

photo = tk.PhotoImage(file = "pink.png")
imageLabel = tk.Label(frame1,image=photo)
imageLabel.pack()

theButton = tk.Button(frame2,
                    text = "我已满18岁！",
                    command = callback)
theButton.pack()

frame1.pack(padx = 10,pady = 10)
frame2.pack(padx = 10,pady = 10)

imageLabel.pack()
root.mainloop()
