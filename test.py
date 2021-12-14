# dic={1:"123","name":"zhangsan","height":180}
# print(dic)
#
# dic['name']="wangwu"
# print(dic)
#
# del dic['name']
# print(dic)
#
# dic["age"]=18
# print(dic)

import turtle

turtle.pensize(5)
turtle.pencolor("yellow")
turtle.fillcolor("red")

for i in range(0,4):
    turtle.forward(100)
    turtle.left(90)

    # coding=utf-8
# import turtle
# import time
#
    # 同时设置pencolor=color1, fillcolor=color2
turtle.color("red", "yellow")

turtle.begin_fill()
for _ in range(50):
    turtle.forward(200)
    turtle.left(170)
    turtle.end_fill()


turtle.mainloop()