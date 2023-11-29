import pathlib
from itertools import count

import numpy as np
import copy

num_nodes_per_cell = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "quad8": 8,
    "tetra": 4,
    "hexahedron": 8,
    "hexahedron20": 20,
    "hexahedron24": 24,
    "wedge": 6,
    "pyramid": 5,
    #
    "line3": 3,
    "triangle6": 6,
    "quad9": 9,
    "tetra10": 10,
    "hexahedron27": 27,
    "wedge15": 15,
    "wedge18": 18,
    "pyramid13": 13,
    "pyramid14": 14,
    #
    "line4": 4,
    "triangle10": 10,
    "quad16": 16,
    "tetra20": 20,
    "wedge40": 40,
    "hexahedron64": 64,
    #
    "line5": 5,
    "triangle15": 15,
    "quad25": 25,
    "tetra35": 35,
    "wedge75": 75,
    "hexahedron125": 125,
    #
    "line6": 6,
    "triangle21": 21,
    "quad36": 36,
    "tetra56": 56,
    "wedge126": 126,
    "hexahedron216": 216,
    #
    "line7": 7,
    "triangle28": 28,
    "quad49": 49,
    "tetra84": 84,
    "wedge196": 196,
    "hexahedron343": 343,
    #
    "line8": 8,
    "triangle36": 36,
    "quad64": 64,
    "tetra120": 120,
    "wedge288": 288,
    "hexahedron512": 512,
    #
    "line9": 9,
    "triangle45": 45,
    "quad81": 81,
    "tetra165": 165,
    "wedge405": 405,
    "hexahedron729": 729,
    #
    "line10": 10,
    "triangle55": 55,
    "quad100": 100,
    "tetra220": 220,
    "wedge550": 550,
    "hexahedron1000": 1000,
    "hexahedron1331": 1331,
    #
    "line11": 11,
    "triangle66": 66,
    "quad121": 121,
    "tetra286": 286,
}

_topological_dimension = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "hexahedron": 3,
    "wedge": 3,
    "pyramid": 3,
    "line3": 1,
    "triangle6": 2,
    "quad9": 2,
    "tetra10": 3,
    "hexahedron27": 3,
    "wedge18": 3,
    "pyramid14": 3,
    "vertex": 0,
    "quad8": 2,
    "hexahedron20": 3,
    "triangle10": 2,
    "triangle15": 2,
    "triangle21": 2,
    "line4": 1,
    "line5": 1,
    "line6": 1,
    "tetra20": 3,
    "tetra35": 3,
    "tetra56": 3,
    "quad16": 2,
    "quad25": 2,
    "quad36": 2,
    "triangle28": 2,
    "triangle36": 2,
    "triangle45": 2,
    "triangle55": 2,
    "triangle66": 2,
    "quad49": 2,
    "quad64": 2,
    "quad81": 2,
    "quad100": 2,
    "quad121": 2,
    "line7": 1,
    "line8": 1,
    "line9": 1,
    "line10": 1,
    "line11": 1,
    "tetra84": 3,
    "tetra120": 3,
    "tetra165": 3,
    "tetra220": 3,
    "tetra286": 3,
    "wedge40": 3,
    "wedge75": 3,
    "hexahedron64": 3,
    "hexahedron125": 3,
    "hexahedron216": 3,
    "hexahedron343": 3,
    "hexahedron512": 3,
    "hexahedron729": 3,
    "hexahedron1000": 3,
    "wedge126": 3,
    "wedge196": 3,
    "wedge288": 3,
    "wedge405": 3,
    "wedge550": 3,
}

abaqus_to_meshio_type = {
    # trusses
    "T2D2": "line",
    "T2D2H": "line",
    "T2D3": "line3",
    "T2D3H": "line3",
    "T3D2": "line",
    "T3D2H": "line",
    "T3D3": "line3",
    "T3D3H": "line3",
    # beams
    "B21": "line",
    "B21H": "line",
    "B22": "line3",
    "B22H": "line3",
    "B31": "line",
    "B31H": "line",
    "B32": "line3",
    "B32H": "line3",
    "B33": "line3",
    "B33H": "line3",
    # surfaces
    "CPS4": "quad",
    "CPS4R": "quad",
    "S4": "quad",
    "S4R": "quad",
    "S4RS": "quad",
    "S4RSW": "quad",
    "S4R5": "quad",
    "S8R": "quad8",
    "S8R5": "quad8",
    "S9R5": "quad9",
    # "QUAD": "quad",
    # "QUAD4": "quad",
    # "QUAD5": "quad5",
    # "QUAD8": "quad8",
    # "QUAD9": "quad9",
    #
    "CPS3": "triangle",
    "STRI3": "triangle",
    "S3": "triangle",
    "S3R": "triangle",
    "S3RS": "triangle",
    # "TRI7": "triangle7",
    # 'TRISHELL': 'triangle',
    # 'TRISHELL3': 'triangle',
    # 'TRISHELL7': 'triangle',
    #
    "STRI65": "triangle6",
    # 'TRISHELL6': 'triangle6',
    # volumes
    "C3D8": "hexahedron",
    "C3D8H": "hexahedron",
    "C3D8I": "hexahedron",
    "C3D8IH": "hexahedron",
    "C3D8R": "hexahedron",
    "C3D8RH": "hexahedron",
    # "HEX9": "hexahedron9",
    "C3D20": "hexahedron20",
    "C3D20H": "hexahedron20",
    "C3D20R": "hexahedron20",
    "C3D20RH": "hexahedron20",
    # "HEX27": "hexahedron27",
    #
    "C3D4": "tetra",
    "C3D4H": "tetra4",
    # "TETRA8": "tetra8",
    "C3D10": "tetra10",
    "C3D10H": "tetra10",
    "C3D10I": "tetra10",
    "C3D10M": "tetra10",
    "C3D10MH": "tetra10",
    # "TETRA14": "tetra14",
    #
    # "PYRAMID": "pyramid",
    "C3D6": "wedge",
    "C3D15": "wedge15",
    #
    # 4-node bilinear displacement and pore pressure
    "CAX4P": "quad",
}
meshio_to_abaqus_type = {v: k for k, v in abaqus_to_meshio_type.items()}
# print(meshio_to_abaqus_type)

class ReadError(Exception):
    pass

class WriteError(Exception):
    pass

def read(filename):
    """Reads a Abaqus inp file."""
    with open(filename, "r") as f:
        out = read_buffer(f)
    return out

def read_buffer(f):
    # Initialize the optional data fields
    points = []
    cells = []
    cell_ids = []
    point_sets = {}
    cell_sets = {}
    cell_sets_element = {}  # Handle cell sets defined in ELEMENT
    cell_sets_types_element = {}
    cell_sets_element_order = []  # Order of keys is not preserved in Python 3.5
    field_data = {}
    cell_data = {}
    point_data = {}
    point_ids = None
    id_cells= {}
    point_ids_sets = {}
    cell_ids_sets = {}
    cell_types_element = {}
    
    line = f.readline()
    while True:
        if not line:  # EOF
            break

        # Comments
        if line.startswith("**"):
            line = f.readline()
            continue

        keyword = line.partition(",")[0].strip().replace("*", "").upper()    
        if keyword == "NODE":
            points, point_ids, line = _read_nodes(f) 
            point_ids_sets = dict(zip(point_ids, points))
            # print(points, point_ids, point_ids_sets) 
            
        elif keyword == "ELEMENT":
            if point_ids is None:
                raise ReadError("Expected NODE before ELEMENT") 
              
            params_map = get_param_map(line, required_keys=["TYPE"]) 
            cell_type, cells_data, ids, sets, line = _read_cells(
                f, params_map, point_ids
            )
            cell_sets.update(sets)     
    
        else:
            # There are just too many Abaqus keywords to explicitly skip them.
            line = f.readline()
    
    key_sets = []
    key_types = []
    for key, values in cell_sets.items():
        ikey = key.split("&")
        if ikey[0] in key_sets:
            cell_sets_types_element[ikey[0]].update({ikey[1]:copy.deepcopy(values)})
            cell_sets_element[ikey[0]].update(copy.deepcopy(values))
        else:
            cell_sets_types_element[ikey[0]] = {ikey[1]:copy.deepcopy(values)}
            cell_sets_element[ikey[0]] = copy.deepcopy(values)
        key_sets.append(ikey[0])
        
        if ikey[1] in key_types:
            cell_types_element[ikey[1]].update(copy.deepcopy(values))
        else:
            cell_types_element[ikey[1]] = copy.deepcopy(values)
        key_types.append(ikey[1])
        
        for subkey, subvalues in copy.deepcopy(values).items():
            cell_ids_sets.update({subkey:subvalues})
        
    # print(len(list(cell_types_element["wedge"].keys())))
    # print(len(list(cell_sets_element["gz"].keys())))
    # print(cell_sets_types_element["gz1"])

    return point_ids_sets, cell_ids_sets, cell_sets_element, cell_types_element, cell_sets_types_element    
            
def _read_nodes(f):
    points = []
    point_ids = []
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        if line.strip() == "":
            continue

        line = line.strip().split(",")
        point_id, coords = line[0], line[1:]
        point_ids.append(int(point_id))
        points.append([float(x) for x in coords])

    return np.array(points, dtype=float), np.array(point_ids, dtype=int), line  

def _read_cells(f, params_map, point_ids):
    etype = params_map["TYPE"]
    if etype not in abaqus_to_meshio_type.keys():
        raise ReadError(f"Element type not available: {etype}")

    cell_type = abaqus_to_meshio_type[etype]
    # ElementID + NodesIDs
    num_data = num_nodes_per_cell[cell_type] + 1

    idx = []
    while True:
        line = f.readline()
        if not line or line.startswith("*"):
            break
        line = line.strip()
        if line == "":
            continue
        idx += [int(k) for k in filter(None, line.split(","))]

    # Check for expected number of data,% 求余运算
    if len(idx) % num_data != 0:
        raise ReadError("Expected number of data items does not match element type")

    idx = np.array(idx).reshape((-1, num_data))
    cell_ids = idx[:, 0]
    cells = idx[:, 1:]
    id_cell = dict(zip(cell_ids , cells))
    
    

    cell_sets = (
        # {params_map["ELSET"]: idx[:, 0]}
        {params_map["ELSET"]+"&"+cell_type: id_cell}
        if "ELSET" in params_map.keys()
        else {}
    )
    
    cell_type = params_map["ELSET"]+"&"+cell_type

    return cell_type, cells, cell_ids, cell_sets, line #cell_type, cells_data, ids, sets, line

def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> word = 'elset,instance=dummy2,generate'
    >>> params = get_param_map(word, required_keys=['instance'])
    params = {
        'elset' : None,
        'instance' : 'dummy2,
        'generate' : None,
    }
    """
    if required_keys is None:
        required_keys = []
    words = word.split(",")
    param_map = {}
    for wordi in words:
        if "=" not in wordi:
            key = wordi.strip().upper()
            value = None
        else:
            sword = wordi.split("=")
            if len(sword) != 2:
                raise ReadError(sword)
            key = sword[0].strip().upper()
            value = sword[1].strip()
        param_map[key] = value

    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += f"{key} not found in {word}\n"
    if msg:
        raise RuntimeError(msg)
    return param_map
    


if __name__ == "__main__":
    inp_path = r"C:\Users\Gj\Desktop\flle3d.inp"
    shapes = read(inp_path)
    print("节点号：", shapes[0].keys())
    print("节点坐标：", shapes[0].values())
    print("单元号：", shapes[1].keys())
    print("单元组成：", shapes[1].values())
    print("单元材料分类：", shapes[2].keys())
    print("单元类型分类：", shapes[3].keys())
    print("单元材料分类+单元类型分类 :", shapes[4]["gz1"])
    

