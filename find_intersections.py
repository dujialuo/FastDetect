from utils import xp, to_device, to_host, to_scalar, wrap, pltfig1, pltfig
from Config import Config
from reader import SlidingComplex64Reader
from math import ceil, floor
import plotly.graph_objects as go

def find_intersections(coefa: xp.ndarray, coefb_in: xp.ndarray, tstart2: float, tdiff: float, reader: SlidingComplex64Reader, epsilon: float, pidx: int, draw: bool):
    """
    Finds all intersection points of two quadratic polynomials within a specified range.

    Args:
        coefa (xp.ndarray): Coefficients of the first quadratic polynomial [c2, c1, c0].
        coefb (xp.ndarray): Coefficients of the second quadratic polynomial [c2, c1, c0].
        tstart2 (float): The center of the search range.
        epsilon (float): The half-width of the search range.

    Returns:
        xp.ndarray: A sorted array of the x-coordinates of the intersection points.
    """
### tdiff: coefb shift right by tdiff
    coefb = xp.copy(coefb_in)
    coefb[1] -= 2 * coefb[0] * tdiff
    coefb[2] -= xp.polyval(coefb, tstart2 + tdiff) - xp.polyval(coefb_in, tstart2)
    print(f"{xp.polyval(coefb, tstart2 + tdiff)=}, {xp.polyval(coefb_in, tstart2)=} {-coefa[1]/2/coefa[0]=} {-coefb[1]/2/coefb[0]=} {tdiff=}")

    x_min = tstart2 - epsilon
    x_max = tstart2 + epsilon
    print(f"{coefa=}, {coefb=}, {tstart2=}, {tdiff=}, {epsilon=} {x_min=}, {x_max=}")

    # Compute the difference polynomial coefa - coefb
    poly_diff = xp.polysub(coefa, coefb)

    # The difference polynomial is also a quadratic: poly_diff(x) = ax^2 + bx + c
    a, b, c = poly_diff

    # Find the vertex of the difference polynomial, if it exists within the range
    if a != 0:
        x_vertex = -b / (2 * a)
        y_vertex = xp.polyval(poly_diff, x_vertex)
    else:  # The difference is a linear function
        x_vertex = None
        y_vertex = None
    y_min_bound = xp.polyval(poly_diff, x_min)
    y_max_bound = xp.polyval(poly_diff, x_max)

    # Determine the range of y-values for the difference polynomial within [x_min, x_max]
    y_values = [y_min_bound, y_max_bound]
    if x_vertex is not None and x_min <= x_vertex <= x_max:
        y_values.append(y_vertex)

    y_lower = min(y_values)
    y_upper = max(y_values)

    n_min = ceil((y_lower - xp.pi) / (2 * xp.pi))
    n_max = floor((y_upper - xp.pi) / (2 * xp.pi))

    nrange = xp.arange(int(n_min), int(n_max) + 1) * 2 * xp.pi

    roots = []
    if a == 0 and b == 0:  
        raise Exception("The two polynomials are identical; infinite intersections.")
        return xp.array([])
    elif a == 0:  
        for n in nrange:
            root = (n - c) / b
            if x_min <= root <= x_max:
                roots.append(root)
    else:
        roots = []
        for n in nrange:
            shifted_c = c - n
            discriminant = b**2 - 4 * a * shifted_c
            if discriminant >= 0:
                sqrt_discriminant = xp.sqrt(discriminant)
                root1 = (-b + sqrt_discriminant) / (2 * a)
                root2 = (-b - sqrt_discriminant) / (2 * a)
                if x_min <= root1 <= x_max:
                    roots.append(root1)
                if x_min <= root2 <= x_max:
                    roots.append(root2)
    intersection_points = xp.array(roots)

    xv = to_device(xp.arange(ceil(x_min), ceil(x_max), dtype=int))
    sig = reader.get(ceil(x_min), ceil(x_max))
    sig_ref1 = sig * xp.exp(-1j * xp.polyval(coefa, xv - (pidx - 1) * Config.tsign))
    sig_ref2 = sig * xp.exp(-1j * xp.polyval(coefb_in, xv - pidx * Config.tsign))
    print(f"{xp.abs(xp.sum(sig_ref1))=}, {xp.abs(xp.sum(sig_ref2))=}")
    

    if len(intersection_points) != 0:
        selected = max(intersection_points, key=lambda x: xp.abs(xp.sum(sig_ref1[:ceil(x - xv[0])])) + xp.abs(xp.sum(sig_ref2[ceil(x - xv[0]):])))
        selected2 = min(intersection_points, key=lambda x: abs(x - tstart2))

    if draw:
        x_vals = xp.linspace(x_min, x_max, 400)
        y_vals_a = wrap(xp.polyval(coefa, x_vals  - (pidx-1) * Config.tsign))
        y_vals_b = wrap(xp.polyval(coefb, x_vals  - (pidx-1) * Config.tsign))
        fig = pltfig1(x_vals, y_vals_a)
        pltfig1(x_vals, y_vals_b, fig=fig)
        pltfig1(xv, xp.angle(sig), fig=fig, mode='markers', marker=dict(color='green', symbol='x', size=8))
        fig.add_trace(go.Scatter(x=to_host(intersection_points), y=to_host(wrap(xp.polyval(coefa, intersection_points  - (pidx-1) * Config.tsign))), mode='markers', marker=dict(color='red', symbol='circle', size=10)))
        fig.add_trace(go.Scatter(x=to_host(intersection_points), y=to_host(wrap(xp.polyval(coefb, intersection_points  - (pidx-1) * Config.tsign))), mode='markers', marker=dict(color='red', symbol='circle', size=10)))
        if len(intersection_points) != 0:
            fig.add_trace(go.Scatter(x=[to_scalar(selected)], y=[to_scalar(wrap(xp.polyval(coefa, selected)))], mode='markers', marker=dict(color='blue', symbol='cross', size=10)))
        fig.add_vline(x=to_scalar(x_min), line_dash='dash')
        fig.add_vline(x=to_scalar(x_max), line_dash='dash')
        fig.add_vline(x=to_scalar(tstart2), line_dash='dash')
        fig.update_layout(title_text=f'Intersection Points of Two Quadratic Polynomials {pidx=}')
        fig.show()
        x_vals = xp.linspace(x_min - Config.tsign, x_max + Config.tsign, 400)
        y_vals_a = wrap(xp.polyval(coefa, x_vals))
        y_vals_b = wrap(xp.polyval(coefb, x_vals))
        pltfig(((x_vals, xp.polyval(coefa, x_vals - (pidx-1) * Config.tsign) - xp.polyval(coefa, tstart2 - (pidx-1) * Config.tsign)), 
                (x_vals, xp.polyval(coefb, x_vals - (pidx-1) * Config.tsign) - xp.polyval(coefb, tstart2 - (pidx-1) * Config.tsign)),
                (x_vals, xp.polyval(coefb_in, x_vals - pidx * Config.tsign) - xp.polyval(coefb_in, tstart2 - pidx * Config.tsign)),
                (xp.arange(ceil(x_min - Config.tsign), ceil(x_max + Config.tsign)), xp.unwrap(xp.angle(reader.get(ceil(x_min - Config.tsign), ceil(x_max + Config.tsign)))))
                ),
                title=f"Intersection Points of Two Quadratic Polynomials {pidx=} {x_vals[0] - ((pidx-1) * Config.tsign)} {x_vals[-1] - (pidx+1)*Config.tsign}", addvline=[tstart2, ],
                modes='lines').show()
        
    if len(intersection_points) == 0:
        raise Exception("No intersection points found within the specified range.")
        return None

    if selected2 != selected:
        print(f"find_intersections(): break point not closeset to tstart2 selected {selected - tstart2 =} {pidx=}")
    return selected
