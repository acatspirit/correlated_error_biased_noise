import dataclasses
import drawsvg as draw
import numpy as np
from CompassCodes import CompassCode


@dataclasses.dataclass
class Point:
    x: float
    y: float

    def __iter__(self):
        return iter([self.x, self.y])

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __truediv__(self, other):
        return Point(self.x / other, self.y/ other)


def z_stabilizers(
    code: CompassCode, d: draw.Drawing, color_z, q_vertices, vertex_radius
):
    hz = np.array(code.H["Z"].todense())

    for sz in hz:
        stroke = "None"
        fill = color_z
        qs = qubits_in(sz)
        if len(qs) == 2:
            pass
            # # we know that these are horizontal
            # h = q_vertices[qs[1]].y - q_vertices[qs[0]].y
            # w = vertex_radius /3*2

            # r = draw.Rectangle(
            #     q_vertices[qs[0]].x - w / 2,
            #     q_vertices[qs[0]].y,
            #     w,
            #     h,
            #     stroke="None",
            #     fill=color_z,
            #     fill_opacity=1,
            # )

            # d.append(r)

        else:
            p = draw.Path(
                stroke=stroke,
                fill=fill,
                fill_opacity=0.6,
            )
            pos = q_vertices[qs[0]]
            p.M(*pos)
            path_order = [q_vertices[qs[q]] for q in range(len(qs) // 2)]
            path_order += [
                q_vertices[qs[q]] for q in reversed(list(range(len(qs) // 2, len(qs))))
            ]
            for next_pos in path_order:
                p.L(*next_pos)

            d.append(p)


def z_stabilizers_just2(
    code: CompassCode, d: draw.Drawing, color_z, q_vertices, vertex_radius
):
    hz = np.array(code.H["Z"].todense())

    for sz in hz:
        stroke = "None"
        fill = color_z
        qs = qubits_in(sz)
        if len(qs) == 2:
            # we know that these are horizontal
            h = q_vertices[qs[1]].y - q_vertices[qs[0]].y
            w = vertex_radius 

            r = draw.Rectangle(
                q_vertices[qs[0]].x - w / 2,
                q_vertices[qs[0]].y,
                w,
                h,
                stroke="None",
                fill=color_z,
                fill_opacity=0.8,
            )

            d.append(r)



def qubits_in(gen):
    qs = np.argwhere(gen == 1)
    qs = qs.reshape(len(qs))
    return qs 


def generator_center(gen, q_vertices):
    # Finds location of center of stabilizer generator gen
    # gen is the row from the parity matrix corresponding to the stab
    vertices = [q_vertices[q] for q in qubits_in(gen)]
    sum = np.sum(vertices)
    sum /= len(vertices)
    return sum

def x_stabilizers(
    code: CompassCode, d: draw.Drawing, color_x, q_vertices, vertex_radius
):
    hx = np.array(code.H["X"].todense())
    for sx in hx:
        qs = qubits_in(sx)
        if len(qs) == 2:
            # we know that these are vertical
            w = q_vertices[qs[1]].x - q_vertices[qs[0]].x
            h = vertex_radius 

            r = draw.Rectangle(
                q_vertices[qs[0]].x,
                q_vertices[qs[0]].y - h / 2,
                w,
                h,
                stroke="None",
                fill=color_x,
                fill_opacity=1,
            )

            d.append(r)

        else:
            p = draw.Path(
                stroke="None",
                fill=color_x,
                fill_opacity=0.70,
            )
            pos = q_vertices[qs[0]]
            p.M(*pos)
            path_order = [q_vertices[qs[q]] for q in range(0, len(qs), 2)]
            path_order += [
                q_vertices[qs[q]] for q in reversed(list(range(1, len(qs), 2)))
            ]
            for next_pos in path_order:
                p.L(*next_pos)

            d.append(p)


def vertices(d, q_vertices, vertex_radius, color_vertex, CD_data):
    print("size, ", vertex_radius+5)
    for q, center in enumerate(q_vertices):
        print(q)
        if CD_data[q] == 0 :
            pass
            d.append(
                draw.Circle(
                    *center,
                    vertex_radius*1.3,
                    fill="darkslategray",
                )
            )
            label = f"{q}"
        elif CD_data[q] == 2:
            print("Hadamard on q ", q)
            d.append(
                draw.Circle(
                    *center,
                    vertex_radius*1.3,
                    fill="gold",
                )
            )
            # label = f"{q}"
        # d.append(
        #     draw.Text(
        #         label,
        #         18,
        #         center.x,
        #         center.y,
        #         center=True,
        #         stroke="black",
        #     )
        # )

#####################------ TEST
def CD_MatchingGraph(d, code, CD_data, q_vertices, type, weight = "all", high_weight = 'no'): 
    if type == "both":
        stabs = ["X","Z"]
    else: stabs = [type]

    if weight == "all":
        weights =[1,2] # both high and low weights
    elif weight == "low":
        weights = [1] # Only low weight
    elif weight == "high":
        weights = [2] # Only high weight
            
    if 2 in weights and high_weight == "dashed":
        dash = "13"
    else: 
        dash = "0"
    sw = 7
    # Draw edges
    for stab in stabs: 
        # h = np.array(code.H["X"].todense())
        h = np.array(code.H[stab].todense())
        for g in h:
            for q in qubits_in(g):
                if CD_data[q] == 0:
                    # Low-weight in X 
                    if stab == "X" and 1 in weights:
                        d.append(draw.Line(*generator_center(g, q_vertices),
                                        *q_vertices[q], stroke_width = sw, stroke = 'black'))
                        d.append(draw.Circle(*generator_center(g, q_vertices), 14, fill="black"))
                    elif stab == "Z" and 2 in weights: # High-weight in Z:
                        d.append(draw.Circle(*generator_center(g, q_vertices), 14, fill="black"))
                        d.append(draw.Line(*generator_center(g, q_vertices),
                                        *q_vertices[q], stroke_width = sw, stroke = 'black', stroke_dasharray = dash))
                elif CD_data[q] == 2:
                    # Low-weight in Z
                    if stab == "Z" and 1 in weights:
                        d.append(draw.Circle(*generator_center(g, q_vertices), 14, fill="black"))
                        d.append(draw.Line(*generator_center(g, q_vertices),
                                            *q_vertices[q], stroke_width = sw, stroke = 'black'))
                    elif stab == "X" and 2 in weights: # High-weight in X
                        d.append(draw.Circle(*generator_center(g, q_vertices), 14, fill="black"))                        
                        d.append(draw.Line(*generator_center(g, q_vertices),
                                        *q_vertices[q], stroke_width = sw, stroke = 'black', stroke_dasharray = dash))
                        
#####################------ TEST

def matching_graph(d, h, q_vertices, type, weight = "all", high_weight = 'no'):
    '''
    type: "X", "Z", or "both".
    '''
    for g in h:
        d.append(draw.Circle(
                *generator_center(g, q_vertices),
                14,
                fill="black",
            ))
        if weight == "high" and high_weight == "dashed":
            dash = "10"
        else: 
            dash = "0"
            
        sw = 20
        # Draw edges of generator to support qubits
        if weight == "all":
            if high_weight == 'dashed':                    
                for q in qubits_in(g):
                    if type == "Z":
                        if q%2 == 0:
                            d.append(draw.Line(*generator_center(g, q_vertices),
                        *q_vertices[q], stroke_width = sw, stroke = 'black', stroke_dasharray = "10"))
                        else:
                            d.append(draw.Line(*generator_center(g, q_vertices),
                        *q_vertices[q], stroke_width = sw, stroke = 'black'))
                    elif type == "X":
                        if q%2 == 0:
                            d.append(draw.Line(*generator_center(g, q_vertices),
                        *q_vertices[q], stroke_width = sw, stroke = 'black'))
                        else:
                            d.append(draw.Line(*generator_center(g, q_vertices),
                        *q_vertices[q], stroke_width = sw, stroke = 'black', stroke_dasharray = "10"))
            else: 
                for q in qubits_in(g):
                    d.append(draw.Line(*generator_center(g, q_vertices),
                *q_vertices[q], stroke_width = sw, stroke = 'black'))
   
        elif weight == "low" and type == "X" or weight == "high" and type == "Z":
            # Even qubits
            for q in qubits_in(g):
                if q%2 == 0:                        
                    d.append(draw.Line(*generator_center(g, q_vertices),
                *q_vertices[q], stroke_width = sw, stroke = 'black', stroke_dasharray = dash))
        elif weight == "low"  and type == "Z" or weight == "high" and type == "X":
            # Odd qubits
            for q in qubits_in(g):
                if q%2 == 1:
                    d.append(draw.Line(*generator_center(g, q_vertices),
                *q_vertices[q], stroke_width = sw, stroke = 'black', stroke_dasharray = dash))


def stabilizers_svg(code: CompassCode, filename, CD_data, edges = "all", type = "X", high_weight = 'no'):
    l = int(np.sqrt(code.H["X"].shape[1]))

    d = draw.Drawing(l * 650 / 5, l * 650 / 5, style="background-color:white")

    h_spacing = 120
    v_spacing = 120
    top = 50
    left = 50

    vertex_radius = 15
    color_vertex = "grey"

    color_x = "cornflowerblue"
    color_z = "firebrick"

    q_vertices = []

    for r in range(l):
        for c in range(l):
            center = Point(
                left + c * h_spacing ,
                top + r * v_spacing,
            )
            q_vertices.append(center)
    
    z_stabilizers(code, d, color_z, q_vertices, vertex_radius)
    x_stabilizers(code, d, color_x, q_vertices, vertex_radius)
    z_stabilizers_just2(code, d, color_z, q_vertices, vertex_radius)

    # if type == "both":
    #     matching_graph(d, np.array(code.H["X"].todense()), q_vertices, type = "X", weight = edges, high_weight = high_weight)
    #     matching_graph(d, np.array(code.H["Z"].todense()), q_vertices, type = "Z", weight = edges, high_weight = high_weight)
    # else:
    #     matching_graph(d, np.array(code.H[type].todense()), q_vertices, type = type, weight = edges, high_weight = high_weight)
    ######### TEST
    # CD_MatchingGraph(d, code, CD_data, q_vertices, type, weight = edges, high_weight = high_weight)
    ######### TEST
    # vertices(d, q_vertices, 10, color_vertex, CD_data) # qubits at vertices
    
    with open(filename, "w") as f:
        d.as_svg(f)
        # d.save_png(f)

def CompassModel(code, filename):
    l = int(np.sqrt(code.H["X"].shape[1]))

    d = draw.Drawing(l * 650 / 5, l * 650 / 5, style="background-color:white")

    h_spacing = 120
    v_spacing = 120
    top = 50
    left = 50

    vertex_radius = 15
    color_vertex = "black"

    color_x = "cornflowerblue"
    color_z = "firebrick"

    q_vertices = []
    
    for r in range(l):
        for c in range(l):
            center = Point(
                left + c * h_spacing ,
                top + r * v_spacing,
            )
            q_vertices.append(center)
     
    
    for i in range(l):
        for j in range(l-1):
            # X stabilizers
            d.append(draw.Line(*q_vertices[l*i+j],
                *q_vertices[l*i+j+1], stroke_width = 10, stroke = color_x))
            # Z stabilizers
            d.append(draw.Line(*q_vertices[j*l+i],
                *q_vertices[(j+1)*l+i], stroke_width = 10, stroke = color_z))
    
    # vertices(d, q_vertices, vertex_radius, color_vertex)
     
    with open(filename, "w") as f:
        d.as_svg(f)   
   
if __name__ == "__main__":
    L = 7
    ell = 4
    code = CompassCode(L, l=ell)
    
    CD_data = np.zeros(L**2)
    CDtype= "I"
    if CDtype =="XZZX": 
        #XZZX 
        for i in range(L**2):
            if i%2 == 1: 
                CD_data[i] = 2 
    elif CDtype == "XZZXonSqu":
    # XZZX on squares
        for i in np.arange(0,L-1):
            for j in np.arange(0,L-1):
                if (i-j)%ell == 0:
                    # (i,j)
                    # p1 = i*(L) + j
                    # (i+1,j)
                    p2 = (i+1)*(L) + j
                    # (i,j+1)
                    p3 = i*(L) + j + 1
                    # (i+1,j+1)
                    # p4 = (i+1)*(L) + j + 1 
                    CD_data[p2], CD_data[p3]= 2,2 # Applying a Hadamard
    elif CDtype == "ZXXZonSqu":
    # ZXXZ on squares
        for i in np.arange(0,L-1):
            for j in np.arange(0,L-1):
                if (i-j)%ell == 0:
                    # (i,j)
                    p1 = i*(L) + j
                    p4 = (i+1)*(L) + j + 1 
                    CD_data[p1], CD_data[p4]= 2,2 # Applying a Hadamard
        if (L-1)%ell == 0:
            CD_data[L-1] = 2
            CD_data[L**2-L] = 2
    fname = "test.svg"
    # fname = f"FinalPlots/MatchingGraphs/PDFS/Graph_l{ell}_Xstabs_{CDtype}.svg"
    # fname = f"FinalPlots/MatchingGraphs/PDFS/Graph_l{ell}_Zstabs_{CDtype}.svg"
    # fname = f"FinalPlots/MatchingGraphs/l{ell}_lowweight_{CDtype}.svg"
    stabilizers_svg(code, fname, CD_data, edges = "low", type = "both", high_weight = 'dashed')
    # fname = f"FinalPlots/MatchingGraphs/qbits_l{ell}_lowweight_{CDtype}_mod{L%6}.svg"
    # stabilizers_svg(code, fname, CD_data, edges = "low", type = "both", high_weight = 'dashed')
    # CompassModel(code, fname)
    # stabilizers_svg(code, fname, edges = "high", stabs = "both", high_weight = 'dashed')
    print("result in test.svg")
