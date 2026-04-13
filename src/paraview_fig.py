"""ParaView outputter that renders PNG images from simulation data."""

#!/usr/bin/env pvpython
import pathlib
import sys

from paraview.simple import *

# Check for correct usage
if len(sys.argv) < 5:
    print(
        "Usage: $PARAVIEW_PYTHON paraview_fig.py <paraview_file_name> <field_of_interest> <time_increment> <mesh/no mesh (1/0)> <distortion_scaling> <colorbar scaling (optional)> <paper_mode (optional, 1/0)>"
    )
    sys.exit(1)


def parse_time_range(arg: str, available: list[int]) -> list[int]:
    """Parse a start-end range, single index, or 'all' against available timesteps.

    Parameters
    ----------
    arg : str
        Either 'all', a start-end range like "0-10", or a single index.
    available : list[int]
        List of available timestep indices.

    Returns
    -------
    list[int]
        List of timestep indices to use.
    """
    if arg.lower() == "all":
        return list(range(len(available)))
    if "(" in arg or "[" in arg:
        start_str, end_str = arg.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        start = max(0, min(start, len(available) - 1))
        end = max(0, min(end, len(available) - 1))
        return list(range(start, end + 1))
    idx = int(arg)
    if idx < 0:
        idx = len(available) - 1
    elif idx >= len(available):
        idx = len(available) - 1
    return [idx]


def render_colorbar_only(
    lut: "ColorTransferFunction",
    data_to_show: "PVTrivialProducer",
    field_arg: str,
    output_image_path: pathlib.Path,
    resolution: int,
    data_min: float,
    data_max: float,
) -> None:
    """Render a separate image containing only the colorbar with min/max labels.

    Parameters
    ----------
    lut : ColorTransferFunction
        The color transfer function for the scalar bar.
    data_to_show : PVTrivialProducer
        The data source to associate with the scalar bar.
    field_arg : str
        Field name to color by (e.g., "displacement" or "displacement.x").
    output_image_path : pathlib.Path
        Path where the colorbar image will be saved.
    resolution : int
        Width of the output image in pixels.
    data_min : float
        Minimum value for the colorbar range.
    data_max : float
        Maximum value for the colorbar range.
    """
    colorbar_view = CreateRenderView()
    try:
        colorbar_view.UseOffscreenRendering = 1
    except Exception:
        pass
    try:
        rw = colorbar_view.GetRenderWindow()
        if rw is not None:
            rw.SetOffScreenRendering(1)
    except Exception:
        pass

    # Single Color
    colorbar_view.BackgroundColorMode = "Single Color"
    colorbar_view.UseColorPaletteForBackground = 0
    colorbar_view.Background = [1.0, 1.0, 1.0]
    colorbar_view.OrientationAxesVisibility = 0

    # We need a representation in this view so the scalar bar renders
    cb_display = Show(data_to_show, colorbar_view, "GeometryRepresentation")

    # Color by the same field
    if "." in field_arg:
        field, comp_str = field_arg.split(".", 1)
        try:
            comp_index = int(comp_str)
        except ValueError:
            mapping = {"x": 0, "y": 1, "z": 2}
            comp_index = mapping.get(comp_str.lower(), 0)
        ColorBy(cb_display, ("POINTS", field, comp_index))
    else:
        ColorBy(cb_display, ("POINTS", field_arg))

    cb_display.SetScalarBarVisibility(colorbar_view, True)
    cb_display.Opacity = 0.0

    # Explicitly rescale the LUT to the known data range
    lut.RescaleTransferFunction(data_min, data_max)

    colorbar_view.Update()

    # Configure the scalar bar
    scalarBar = GetScalarBar(lut, colorbar_view)
    scalarBar.Visibility = 1
    scalarBar.WindowLocation = "Any Location"
    scalarBar.Orientation = "Horizontal"
    scalarBar.ScalarBarLength = 0.8
    scalarBar.Position = [0.1, 0.0]

    # No field title, show tick labels for min/max
    scalarBar.Title = ""
    scalarBar.ComponentTitle = ""
    scalarBar.TitleFontSize = 1
    scalarBar.LabelFontSize = 14
    scalarBar.TitleColor = [0.0, 0.0, 0.0]
    scalarBar.LabelColor = [0.0, 0.0, 0.0]
    scalarBar.AddRangeLabels = 1
    scalarBar.DrawTickLabels = 0
    scalarBar.DrawAnnotations = 0
    scalarBar.DrawTickMarks = 0
    scalarBar.AutomaticLabelFormat = 0
    scalarBar.RangeLabelFormat = "%.1f"
    scalarBar.ScalarBarThickness = 40
    scalarBar.Modified()

    Render(colorbar_view)

    # Save — tall enough for labels above the bar
    SaveScreenshot(
        str(output_image_path),
        colorbar_view,
        ImageResolution=[resolution, int(resolution * 0.32)],
    )
    Delete(colorbar_view)


# Read command-line arguments
pvd_file_name = sys.argv[1]
field_arg = sys.argv[2]
time_index = sys.argv[3]
mesh_outline = sys.argv[4]
distortions_scaling = float(sys.argv[5])
scaling_arg = sys.argv[6] if len(sys.argv) > 6 else "none"
paper_mode = sys.argv[7] == "1" if len(sys.argv) > 7 else False

# Assemble full file path and output directory
filename = f"{pvd_file_name}.pvd"
output_path = pathlib.Path(pvd_file_name).resolve().parent.parent

# Open the file
data = OpenDataFile(filename)
UpdatePipeline()

sources = GetSources()
source_entries = list(sources.keys())
source = sources[source_entries[0]]

# Create a WarpByVector filter if distortions are requested
warpFilter = WarpByVector(Input=source)
data_to_show = warpFilter

# Create the render view
renderView = GetActiveViewOrCreate("RenderView")
try:
    # ParaView property (preferred)
    renderView.UseOffscreenRendering = 1
except Exception:
    pass
try:
    # VTK fallback (works across versions)
    rw = renderView.GetRenderWindow()
    if rw is not None:
        rw.SetOffScreenRendering(1)
except Exception:
    pass

display = Show(data_to_show, renderView, "GeometryRepresentation")
display.Representation = "Surface With Edges" if mesh_outline == "1" else "Surface"

# Hide the source mesh to display only the warped mesh
Hide(source, renderView)
warpFilter.Vectors = ["POINTS", "displacement"]
warpFilter.ScaleFactor = distortions_scaling


# Determine if a component was provided (e.g. "displacement.x")
if "." in field_arg:
    field, comp_str = field_arg.split(".", 1)
    try:
        # Try to convert the component to an integer
        comp_index = int(comp_str)
    except ValueError:
        # Map common letter components to indices
        mapping = {"x": 0, "y": 1, "z": 2}
        comp_index = mapping.get(comp_str.lower(), 0)  # default to 0 if unknown
    # Set the color by the field and component
    ColorBy(display, ("POINTS", field, comp_index))
    field_name = field
else:
    # Set the color by the component
    ColorBy(display, ("POINTS", field_arg))
    field_name = field_arg

# Show the color bar and adjust its appearance and position
lut = GetColorTransferFunction(field_name)
lut.ApplyPreset("Turbo", True)
lut.InvertTransferFunction()

if not paper_mode:
    # Standard mode: show colorbar on the main render
    display.SetScalarBarVisibility(renderView, True)
    scalarBar = GetScalarBar(lut, renderView)
    renderView.Update()

    # Format the scalar bar title, position and labels
    if scalarBar:
        scalarBar.WindowLocation = "Any Location"
        scalarBar.Orientation = "Horizontal"
        scalarBar.ScalarBarLength = 0.3
        scalarBar.Position = [(1.0 - scalarBar.ScalarBarLength) / 2.0, 0.02]
        scalarBar.TitleFontSize = 6
        scalarBar.LabelFontSize = 4
        scalarBar.TitleColor = [0.06, 0.06, 0.06]
        scalarBar.LabelColor = [0.06, 0.06, 0.06]
        scalarBar.Modified()
else:
    # Paper mode: hide colorbar from the main render
    display.SetScalarBarVisibility(renderView, False)
    renderView.Update()

# Scale the color transfer function if a scaling argument is provided
scaling_values = None
if scaling_arg not in ("", "none"):
    try:
        # Expecting the tuple in the form "(min,max)" or "min,max"
        scaling_values = tuple(
            map(float, scaling_arg.strip("()").strip("[]").split(","))
        )
        if len(scaling_values) == 2:
            # Rescale the color transfer function using the provided range
            lut.RescaleTransferFunction(scaling_values[0], scaling_values[1])
    except Exception as e:
        print("Error parsing scaling tuple:", e)

# Render the scene
renderView.OrientationAxesVisibility = 0
renderView.ResetCamera()

# In paper mode, zoom in to minimise margins
if paper_mode:
    camera = renderView.GetActiveCamera()
    camera.Dolly(1.4)
    Render(renderView)

times = getattr(data, "TimestepValues", [])
if not times:
    print("No timesteps found, rendering the current view only.")
    indices = [0]
else:
    indices = parse_time_range(time_index, times)

# Ensure plots output directory exists
plots_output_path = output_path / "plots"
plots_output_path.mkdir(parents=True, exist_ok=True)

# Reset the background color and mode
renderView.BackgroundColorMode = "Single Color"
renderView.UseColorPaletteForBackground = 0
renderView.Background = [1.0, 1.0, 1.0]

# Sanitise the paraview file name for output
pvd_file_name = pathlib.Path(pvd_file_name).stem
pvd_file_name = pvd_file_name.replace("/", "_").replace("\\", "_").replace(" ", "_")

# Set resolution based on mode
if paper_mode:
    resolution = 500
else:
    resolution = 1000

for idx in indices:
    # If there are timestep arguments, set the view time
    if times:
        renderView.ViewTime = times[idx]
        renderView.Update()

        # Re-apply zoom after time update in paper mode
        if paper_mode:
            renderView.ResetCamera()
            camera = renderView.GetActiveCamera()
            camera.Dolly(1.4)

    # Update the view
    Render(renderView)

    # Generate the filename for the output image
    output_image = f"pv_render_{pvd_file_name}_{field_arg}_{idx}.png"
    output_image_path = plots_output_path / output_image

    if paper_mode:
        # Square image with tight margins
        SaveScreenshot(
            str(output_image_path),
            renderView,
            ImageResolution=[resolution, resolution],
        )
    else:
        SaveScreenshot(
            str(output_image_path),
            renderView,
            ImageResolution=[resolution, int(resolution * 1.15)],
        )

# In paper mode, render the colorbar as a separate file (once, not per timestep)
if paper_mode:
    # Read the LUT range after all rescaling has been applied
    rgb_pts = lut.RGBPoints
    data_min = rgb_pts[0]
    data_max = rgb_pts[-4]
    print(f"Colorbar range: {data_min} - {data_max}")
    colorbar_output = plots_output_path / "colorbar.png"
    render_colorbar_only(
        lut,
        data_to_show,
        field_arg,
        colorbar_output,
        resolution,
        data_min,
        data_max,
    )
