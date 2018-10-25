#!/usr/bin/python
#-*- coding: utf-8 -*-
import turtle
import math
import tkinter

class Viewer:
    def __init__(self):
        self.l_vessel = 50  # Metade do comprimento da embarcacao
        #first we initialize the turtle settings
        turtle.speed(0)
        turtle.mode('logo')
        turtle.setworldcoordinates(0, -500, 2000, 500)
        turtle.setup()
        turtle.screensize(4000, 1000, 'white')
        w_vessel = 5  # Metade da largura da embarcacao
        turtle.register_shape('vessel', (
            (0, self.l_vessel), (w_vessel, self.l_vessel / 2), (w_vessel, -self.l_vessel), (-w_vessel, -self.l_vessel),
            (-w_vessel, self.l_vessel / 2)))
        turtle.register_shape('rudder', ((-1, 0), (1, 0), (1, -10), (-1, -10)))
        turtle.degrees()

        #
        self.vessel = turtle.Turtle()
        self.vessel.shape('vessel')
        self.vessel.fillcolor('red')
        self.vessel.penup()
        self.rudder = turtle.Turtle()
        self.rudder.shape('rudder')
        self.rudder.fillcolor('green')
        self.rudder.penup()
        self.step_count = 0
        self.steps_for_stamp = 30

    def plot_position(self, x, y, theta, rud_angle):
        converted_angle = theta*180/math.pi #convertion may apply if you use radians
        # turtle.fillcolor('green')
        self.vessel.setpos(x, y)
        self.vessel.setheading(converted_angle)
        self.rudder.setpos(x - self.l_vessel * math.cos(math.pi * converted_angle / 180),
                           y - self.l_vessel * math.sin(math.pi * converted_angle / 180))
        self.rudder.setheading(converted_angle - rud_angle)
        self.vessel.pendown()

    def plot_guidance_line(self, point_a, point_b):
        self.vessel.setpos(point_a[0], point_a[1])
        self.vessel.pendown()
        self.vessel.setpos(point_b[0], point_b[1])
        self.vessel.penup()

    def  plot_goal(self, point, factor):
        turtle.speed(0)
        turtle.setpos(point[0] - factor, point[1] - factor)
        turtle.pendown()
        turtle.fillcolor('red')
        turtle.begin_fill()
        turtle.setpos(point[0] - factor, point[1] + factor)
        turtle.setpos(point[0] + factor, point[1] + factor)
        turtle.setpos(point[0] + factor, point[1] - factor)
        turtle.end_fill()
        turtle.penup()

    def plot_boundary(self, points_list):
        turtle.speed(0)
        turtle.setpos(points_list[0][0], points_list[0][1])
        turtle.pendown()
        turtle.fillcolor('blue')
        turtle.begin_fill()
        for point in points_list:
            turtle.setpos(point[0], point[1])
        turtle.end_fill()
        turtle.penup()

    def freeze_scream(self, ):
        turtle.mainloop()

    def end_episode(self, ):
        self.vessel.penup()
        self.rudder.penup()

    def restart_plot(self):
        self.vessel.pendown()

if __name__ == '__main__':
    viewer = Viewer()
    viewer.plot_guidance_line((0, 0), (500, 0))
    viewer.plot_position(100, 20, 20, 10)
    viewer.freeze_scream()