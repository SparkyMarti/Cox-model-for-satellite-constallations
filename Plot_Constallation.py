import numpy as np
import pyvista as pv
from numba import njit
from matplotlib import cm


def cox_process_plot(lam, mu):
    n_orbits = np.random.poisson(lam)
    if n_orbits == 0:
        return np.array([[0], [0], [np.inf]])

    theta = np.random.uniform(0, np.pi, n_orbits)
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n_orbits))
    n_sats = np.random.poisson(mu, n_orbits)

    total_sats = np.sum(n_sats)
    if total_sats == 0:
        return np.array([[0], [0], [np.inf]])

    valid = n_sats > 0
    theta = np.repeat(theta[valid], n_sats[valid])
    phi = np.repeat(phi[valid], n_sats[valid])

    omega = np.random.uniform(0, 2 * np.pi, total_sats)

    return  theta, phi, omega, n_sats


@njit
def orbital_to_cartesian(r, theta, phi, omega):
    x = r * (np.cos(theta) * np.cos(omega) - np.sin(theta) * np.sin(omega) * np.cos(phi))
    y = r * (np.sin(theta) * np.cos(omega) + np.cos(theta) * np.sin(omega) * np.cos(phi))
    z = r * (np.sin(omega) * np.sin(phi))
    return x, y, z


def plot_system_pyvista(mu, lam, r_earth, r_sat, save=False):
    plotter = pv.Plotter(window_size=(800, 800))

    theta, phi, omega, count = cox_process_plot(mu, lam)

    # Earth as a sphere
    earth = pv.Sphere(radius=r_earth, center=(0, 0, 0), theta_resolution=25, phi_resolution=25)
    plotter.add_mesh(earth,
                     color="white", 
                     smooth_shading=True, 
                     specular=1.0,           # High specular for shininess
                     specular_power=50,      # Sharper highlights for a shiny surface
                     ambient=0.9,            # Ambient light to make it brighter
                     diffuse=0.7)

    # Overlay wireframe grid on Earth
    earth_wire = pv.Sphere(radius=r_earth*1.02, theta_resolution=20, phi_resolution=20)
    plotter.add_mesh(earth_wire, style='wireframe', color='gray', line_width=0.5, opacity=0.5)

    # Setup colormap
    colormap = cm.get_cmap("tab10", len(count))  # One color per orbit group

   # Plot each orbit and its satellites
    omega_vals = np.linspace(0, 2 * np.pi, 100)
    offset = 0
    for i, c in enumerate(count):
        # Satellite color
        rgb_color = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)

        # Plot orbit in black
        orbit_theta = theta[offset]
        orbit_phi = phi[offset]
        x, y, z = orbital_to_cartesian(r_sat, np.array([orbit_theta]), np.array([orbit_phi]), omega_vals)
        orbit_points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        orbit_poly = pv.lines_from_points(orbit_points)
        plotter.add_mesh(orbit_poly, color='black', line_width=2)

        # Plot satellites with group color
        sat_theta = theta[offset:offset + c]
        sat_phi = phi[offset:offset + c]
        sat_omega = omega[offset:offset + c]
        x_s, y_s, z_s = orbital_to_cartesian(r_sat, sat_theta, sat_phi, sat_omega)
        sat_points = np.column_stack((x_s.flatten(), y_s.flatten(), z_s.flatten()))
        sat_poly = pv.PolyData(sat_points)
        sat_poly['colors'] = np.tile(rgb_color, (sat_points.shape[0], 1))
        plotter.add_mesh(sat_poly, scalars='colors', rgb=True, point_size=10, render_points_as_spheres=True,
                        smooth_shading=True, 
                        specular=1.0,           # High specular for shininess
                        specular_power=50,      # Sharper highlights for a shiny surface
                        ambient=0.5,            # Ambient light to make it brighter
                        diffuse=0.7)

        offset += c
    # Plot settings
    plotter.set_background("white")
    plotter.show_bounds(location='outer', xtitle='x [km]',ytitle='y [km]',ztitle='z [km]', bold=False, fmt='%1.0f', font_size=15, axes_ranges=(-7000,7000,-7000,7000,-7000,7000), ticks='outside')
    plotter.view_vector(vector=[2, 0.4, 0.2])
    if save==False:
        plotter.show()  # Keeps the scene open
    else:
       plotter.save_graphic("Used code\Figures\orbit_figure1.pdf")  # Save to PDF
    plotter.close()


plot_system_pyvista(10, 90, 6400,7000)
