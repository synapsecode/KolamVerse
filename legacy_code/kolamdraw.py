# """
# Applying L-System for Kolam pattern Generation: Procedure:
# Axiom or Initiator: FBFBFBFB Rules: A->AFBFA
# B ->AFBFBFBFA From current cursor position ,
# F -> Forward 10 units (draw a line 10 units length) A -> Draw an arc (circle (10, 90))
# B -> Calculate forward units I = 5/sqrt (2)
# Forward I Units
# Draw an arc (circle (I, 270)) Forward I Units
# """

# import argparse
# import turtle
# import tkinter as tk
# from math import sqrt
# from lsystem import generate_lsystem_state

# def draw_kolam(t, state, step=20, angle=0):
#     dot_positions = []

#     t.setheading(angle)
    
#     # Draw the entire pattern and collect dot positions
#     for ch in state:
#         if ch == 'F':
#             t.color('green')
#             t.pensize(3)
#             t.fd(step)
            
#         elif ch == 'A':
#             t.color('blue')
#             t.pensize(3)
            
#             dot_positions.append(t.position())
            
#             t.circle(step * 1.5, 90)  # Quarter-circle arc
            
#         elif ch == 'B':
#             t.color('red')
#             t.pensize(3)
            
#             dot_positions.append(t.position())

#             I = step / sqrt(2)
#             t.fd(I)
#             t.circle(I * 0.7, 270)
#             t.fd(I)
            
#         elif ch == 'L':
#             t.left(45)  # Rotate left 45° without drawing
            
#         elif ch == 'R':
#             t.right(45)  # Rotate right 45° without drawing

#         else:
#             # Unknown symbol → skip
#             pass

#     # Draw the dots at the collected positions
#     unique_dot_positions = set(dot_positions)
    
#     t.penup()
#     t.color('black')
    
#     for x, y in unique_dot_positions:
#         t.goto(x, y)
#         t.dot(6, 'black')
    
#     t.pendown()


# def make_canvas():
#     root = tk.Tk()
#     root.title("Interactive Kolam Generator")

#     frame = tk.Frame(root)
#     frame.pack(fill=tk.BOTH, expand=True)

#     canvas = tk.Canvas(frame, width=800, height=600, scrollregion=(-1000, -1000, 1000, 1000))
#     hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
#     hbar.pack(side=tk.BOTTOM, fill=tk.X)
#     vbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
#     vbar.pack(side=tk.RIGHT, fill=tk.Y)

#     canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
#     canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#     # Turtle setup
#     t = turtle.RawTurtle(canvas)
#     t.speed(0)
#     t.hideturtle()
#     t.pensize(3)
#     t.penup()
#     t.goto(0, 0)  # start at center
#     t.pendown()
#     t.getscreen().tracer(0, 0)  # fast drawing

#     return root, t

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--exact")
#     ap.add_argument("--seed", default="FBFBFBFB")
#     ap.add_argument('--depth', default=1)
#     args = ap.parse_args()
    
#     root, canvas = make_canvas()
#     # L-system input
#     # n = int(tk.simpledialog.askinteger("L-System Kolam", "Enter Level (2-5 recommended):", minvalue=1, maxvalue=6))
#     if(args.exact):
#         state = args.exact
#     else:
#         initial_state = args.seed
#         state = generate_lsystem_state(initial_state, int(args.depth))
#     # draw_kolam(canvas, state, step=20)
#     draw_kolam(canvas, state, step=20, angle=0)
#     canvas.getscreen().update()
#     root.mainloop()

# if __name__ == "__main__":
#     main()