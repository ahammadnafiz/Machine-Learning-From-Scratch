import math
import time
import os

# ANSI color codes for donut-like colors
COLORS = [
    "\033[38;5;220m",  # Light yellow
    "\033[38;5;214m",  # Dark yellow
    "\033[38;5;208m",  # Light brown
    "\033[38;5;130m",  # Medium brown
    "\033[38;5;94m",   # Dark brown
]
RESET = "\033[0m"

def main():
    A, B = 0, 0
    
    while True:
        z = [0] * 3800
        b = [' '] * 3800
        
        for j in range(0, 628, 7):
            for i in range(0, 628, 2):
                c = math.sin(i)
                d = math.cos(j)
                e = math.sin(A)
                f = math.sin(j)
                g = math.cos(A)
                h = d + 2
                D = 1 / (c * h * e + f * g + 5)
                l = math.cos(i)
                m = math.cos(B)
                n = math.sin(B)
                t = c * h * g - f * e
                
                x = int(40 + 30 * D * (l * h * m - t * n))
                y = int(20 + 15 * D * (l * h * n + t * m))
                o = int(x + 80 * y)
                N = int(8 * ((f * e - c * d * g) * m - c * d * e - f * g - l * d * n))
                
                if 47 > y and y > 0 and x > 0 and 79 > x and D > z[o]:
                    z[o] = D
                    b[o] = ".,-~:;=!*#$@"[N if N > 0 else 0]
        
        os.system('cls' if os.name == 'nt' else 'clear')
        for k in range(0, 3800, 80):
            line = ''.join(b[k:k+80])
            colored_line = ''
            for char in line:
                if char != ' ':
                    color = COLORS[int(COLORS.index(COLORS[-1]) * (1 - ".,-~:;=!*#$@".index(char) / 12))]
                    colored_line += f"{color}{char}{RESET}"
                else:
                    colored_line += char
            print(colored_line)
        
        A += 0.04
        B += 0.02
        time.sleep(0.03)

if __name__ == "__main__":
    main()
