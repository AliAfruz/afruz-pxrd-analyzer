"""Offline analytic sphere/cylinder renderer; no GPU, browser or cloud required."""
from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, PngImagePlugin

from .crystal_scene import (
    CrystalScene, ELEMENT_COLORS, SCIENTIFIC_ELEMENT_COLORS, SCIENTIFIC_STYLE,
    MOF_PORE_STYLE, MOF_STYLE, PORE_ORANGE_HEX,
    display_radius, element_color, scene_document,
)
from .crystallography import ATOMIC_NUMBER


def _rgb(color):
    return np.array(ImageColor.getrgb(color), dtype=np.float32) / 255


def _element_color(element, scientific):
    return element_color(element, scientific)


def camera_matrix(azimuth, elevation):
    az, el = np.radians([azimuth, elevation])
    right = np.array([np.cos(az), -np.sin(az), 0])
    up = np.array([np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), np.cos(el)])
    toward = np.cross(right, up)
    return np.array([right, up, toward])


def _font(size, bold=False):
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    for path in (Path("C:/Windows/Fonts") / filename, "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(str(path), max(8, int(size)))
        except OSError:
            pass
    return ImageFont.load_default()


class _Canvas:
    def __init__(self, width, height, light, transparent):
        self.width, self.height = width, height
        y, x = np.ogrid[:height, :width]
        halo = np.exp(-(((x / width - .47) / .59) ** 2 + ((y / height - .4) / .7) ** 2) * 2)
        if light:
            bottom, top = _rgb("#edf0f2"), _rgb("#ffffff")
        else:
            bottom, top = _rgb("#080f1d"), _rgb("#253c54")
        self.rgb = np.asarray(bottom + halo[..., None] * (top - bottom), dtype=np.float32)
        self.alpha = np.full((height, width), 0.0 if transparent else 1.0, dtype=np.float32)
        self.depth = np.full((height, width), -np.inf, dtype=np.float32)
        self.light = light
        self.transparent = transparent

    def patch(self, xmin, xmax, ymin, ymax):
        x0, x1 = max(0, math.floor(xmin)), min(self.width, math.ceil(xmax) + 1)
        y0, y1 = max(0, math.floor(ymin)), min(self.height, math.ceil(ymax) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        return np.s_[y0:y1, x0:x1], xx + .5, yy + .5

    def shade(self, nx, ny, nz, color, visibility=1.0):
        # Analytic surface normals, broad key/fill lights, and a cool rim.
        key = np.maximum(0, -.42 * nx + .58 * ny + .69 * nz)
        fill = np.maximum(0, .72 * nx - .10 * ny + .69 * nz)
        rim = np.maximum(0, .3 * nx + .30 * ny - .905 * nz)
        gloss = np.maximum(0, -.226 * nx + .311 * ny + .923 * nz) ** 65
        broad = np.maximum(0, -.27 * nx + .39 * ny + .88 * nz) ** 9
        illumination = .20 + .60 * key * visibility + .23 * fill
        rgb = color * illumination[..., None]
        rgb += (gloss * visibility)[..., None] * .48 + (broad * visibility)[..., None] * .14
        rgb += rim[..., None] * np.array([.14, .25, .34])
        return np.clip(rgb, 0, 1)

    def write(self, region, depth, mask, colors, opacity=1.0):
        visible = mask & (depth > self.depth[region])
        if not np.any(visible):
            return
        target = self.rgb[region]
        if opacity == 1:
            target[visible] = colors[visible] if np.ndim(colors) == 3 else colors
            self.depth[region][visible] = depth[visible]
            self.alpha[region][visible] = 1
        else:
            # Straight-alpha over, also correct when the background is transparent.
            old_alpha = self.alpha[region][visible]
            new_alpha = opacity + old_alpha * (1 - opacity)
            rgb = colors[visible] if np.ndim(colors) == 3 else colors
            target[visible] = (rgb * opacity + target[visible] * old_alpha[:, None] * (1 - opacity)) / new_alpha[:, None]
            self.alpha[region][visible] = new_alpha

    def sphere(self, center, radius, color, casters=(), opacity=1.0):
        cx, cy, cz = center
        patch = self.patch(cx - radius, cx + radius, cy - radius, cy + radius)
        if patch is None:
            return
        region, x, y = patch
        nx, ny = (x - cx) / radius, (cy - y) / radius
        r2 = nx * nx + ny * ny
        nz = np.sqrt(np.maximum(0, 1 - r2))
        depth = cz + radius * nz
        visibility = np.ones_like(depth)
        # Soft sphere shadows along the broad key light; geometry never moves.
        for other, other_radius in casters:
            vx, vy, vz = other[0] - x, y - other[1], other[2] - depth
            along = -.42 * vx + .58 * vy + .69 * vz
            perpendicular = np.sqrt(np.maximum(0, vx * vx + vy * vy + vz * vz - along * along))
            penumbra = np.maximum(radius * .09, np.maximum(along, 0) * .08)
            coverage = np.clip((other_radius - perpendicular) / penumbra + .5, 0, 1)
            visibility *= 1 - .85 * coverage * (along > 0)
        self.write(
            region,
            depth,
            r2 <= 1,
            self.shade(nx, ny, nz, color, visibility),
            opacity,
        )

    def cylinder(self, a, b, radius, color):
        # Finite analytic cylinder, with sphere end caps: true depth/occlusion.
        delta = b - a
        length = np.linalg.norm(delta)
        if length < 1e-7:
            self.sphere(a, radius, color)
            return
        direction = delta / length
        patch = self.patch(min(a[0], b[0]) - radius, max(a[0], b[0]) + radius,
                           min(a[1], b[1]) - radius, max(a[1], b[1]) + radius)
        if patch is None:
            return
        region, x, y = patch
        dx, dy, dz = direction
        qa = 1 - dz * dz
        if qa > 1e-8:
            wx, wy = x - a[0], y - a[1]
            dot = wx * dx + wy * dy
            qb = -2 * dot * dz
            qc = wx * wx + wy * wy - dot * dot - radius * radius
            disc = qb * qb - 4 * qa * qc
            wz = (-qb + np.sqrt(np.maximum(0, disc))) / (2 * qa)
            along = dot + wz * dz
            mask = (disc >= 0) & (along >= 0) & (along <= length)
            nx, ny, nz = (wx - along * dx) / radius, -(wy - along * dy) / radius, (wz - along * dz) / radius
            self.write(region, a[2] + wz, mask, self.shade(nx, ny, nz, color))
        self.sphere(a, radius, color)
        self.sphere(b, radius, color)

    def dashed_cylinder(self, a, b, radius, color, dash_count=5):
        delta = np.asarray(b) - np.asarray(a)
        for index in range(dash_count):
            start = (index + .12) / dash_count
            end = min((index + .62) / dash_count, 1.0)
            self.cylinder(np.asarray(a) + delta * start, np.asarray(a) + delta * end, radius, color)

    def triangle(self, points, color, opacity):
        a, b, c = points
        patch = self.patch(np.min(points[:, 0]), np.max(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 1]))
        if patch is None:
            return
        region, x, y = patch
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-8:
            return
        u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        w = 1 - u - v
        mask = (u >= 0) & (v >= 0) & (w >= 0)
        self.write(region, u * a[2] + v * b[2] + w * c[2], mask, color, opacity)


def render_crystal(scene: CrystalScene, width=1200, height=900, *, antialias=True) -> Image.Image:
    """Render one frozen scene. Safe to call from a worker thread."""
    if not 160 <= width <= 6000 or not 160 <= height <= 6000 or width * height > 16_000_000:
        raise ValueError("Choose dimensions of 160–6,000 px, with at most 16 megapixels.")
    factor = 2 if antialias and max(width, height) <= 2000 and width * height <= 4_000_000 else 1
    w, h = width * factor, height * factor
    settings = scene.settings
    pore_view = settings.style == MOF_PORE_STYLE
    scientific = settings.style in {SCIENTIFIC_STYLE, MOF_STYLE, MOF_PORE_STYLE}
    light = settings.style.startswith("Publication") or scientific
    canvas = _Canvas(w, h, light, settings.transparent)
    if scientific and not settings.transparent:
        canvas.rgb[:] = _rgb("#ffffff")
    rotation = camera_matrix(settings.azimuth, settings.elevation)
    center = np.asarray(settings.repeats) @ scene.basis / 2
    cam = (scene.positions - center) @ rotation.T
    radii = np.array([display_radius(el, scientific) for el in scene.elements]) * settings.atom_scale
    corners = np.array(list(product(*((0, n) for n in settings.repeats)))) @ scene.basis
    extent_geometry = (
        scene.positions
        if settings.isolate_pore
        else np.vstack([corners, scene.positions])
    )
    extent_points = (extent_geometry - center) @ rotation.T
    extent = np.max(np.abs(extent_points[:, :2]), axis=0) + max(radii) * 1.3
    footer = .03 if scientific else (.14 if settings.caption else .03)
    side = .22 if scientific and settings.caption else .04
    available_width = .94 - side
    scale = min(w * available_width / (2 * extent[0]), h * (1 - footer - .10) / (2 * extent[1])) * settings.zoom
    origin = np.array([w * (side + available_width / 2), h * (.49 - footer / 2), 0])

    def project(points):
        out = (np.asarray(points) - center) @ rotation.T * scale
        out[..., 1] *= -1
        return out + origin

    positions = project(scene.positions)
    large_scene = len(positions) > 2500
    if not light and settings.style.startswith("Studio"):
        canvas.rgb *= np.array([1.05, .93, .85], dtype=np.float32)
    # Subtle grounding shadow. It is a compositional backdrop, not a density map.
    if not settings.transparent and not pore_view:
        y, x = np.ogrid[:h, :w]
        floor_y = min(h * (1 - footer - .015), max(positions[:, 1] + radii * scale) + h * .025)
        center_x = origin[0]
        shadow = np.exp(-((x - center_x) / (w * .24)) ** 2 - ((y - floor_y) / (h * .021)) ** 2)
        canvas.rgb *= 1 - (.10 if scientific else (.13 if light else .30)) * shadow[..., None]
    edge_color = _rgb("#282d31" if scientific else ("#8b9aa9" if light else "#698197"))
    for edge in scene.edges:
        a, b = project(edge)
        canvas.cylinder(a, b, max(.45 * factor, (.009 if scientific else .016) * scale), edge_color)

    # Draw translucent maximal-empty-sphere estimates first so the framework
    # remains crisp in front while the yellow volume supplies a strong 3-D cue.
    projected_pores = [
        (project(center), radius * scale)
        for center, radius in zip(scene.pore_centers, scene.pore_radii)
    ]
    for center, radius in sorted(projected_pores, key=lambda item: float(item[0][2])):
        canvas.sphere(
            center,
            radius,
            _rgb(PORE_ORANGE_HEX),
            opacity=settings.pore_opacity,
        )

    # Polyhedra sit behind atom spheres. The center element controls face color in
    # the scientific preset, allowing Fe and Ni polyhedra in the same figure.
    projected_faces = [(project(face), element) for face, element in zip(scene.triangles, scene.triangle_elements)]
    for triangle, element in sorted(projected_faces, key=lambda item: float(np.mean(item[0][:, 2]))):
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        if pore_view:
            # Element-aware CPK/Jmol-like node colors are scientifically less
            # ambiguous than a decorative single-color MOF template.  For
            # MIL-101(Cr), chromium polyhedra therefore appear lavender-blue.
            color = _rgb(_element_color(element, True)) * (.70 + .30 * abs(normal[2]))
        elif scientific:
            color = _rgb(_element_color(element, True)) * (.72 + .28 * abs(normal[2]))
        else:
            color = _rgb("#a881c0" if light else "#b58ad3") * (.65 + .35 * abs(normal[2]))
        canvas.triangle(triangle, color, settings.polyhedron_opacity)
    if settings.polyhedron_edges:
        for triangle, element in projected_faces:
            color = (
                _rgb(_element_color(element, True)) * .70
                if pore_view
                else _rgb(_element_color(element, scientific)) * .72
            )
            for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
                canvas.cylinder(start, end, max(.35 * factor, .009 * scale), color)

    if settings.hydrogen_bonds:
        for h_index, acceptor_index in scene.hydrogen_bonds:
            canvas.dashed_cylinder(positions[h_index], positions[acceptor_index],
                                   max(.5 * factor, .018 * scale), _rgb("#8c8c8c"), 5)
    if settings.bonds:
        for i, j in scene.bonds:
            a, b = positions[i], positions[j]
            mid = (a + b) / 2
            for start, end, element in ((a, mid, scene.elements[i]), (mid, b, scene.elements[j])):
                color = (
                    _rgb("#515b57")
                    if pore_view
                    else _rgb(_element_color(element, scientific))
                    * (.76 if scientific else .82)
                    + (.10 if scientific else .06)
                )
                canvas.cylinder(start, end, settings.bond_radius * scale, color)
    atom_order = (
        np.argsort(positions[:, 2])
        if settings.atom_opacity < 1.0
        else np.arange(len(positions))
    )
    for index in atom_order:
        position = positions[index]
        radius = radii[index]
        element = scene.elements[index]
        # All-pairs soft-shadow discovery is quadratic for expanded P1 MOFs.
        # Large scenes retain depth-buffered occlusion, but omit inter-sphere
        # cast shadows so a full framework remains practical to render.
        if large_scene:
            casters = ()
        else:
            delta = positions - position
            light_direction = np.array([-.42, -.58, .69])
            along = delta @ light_direction
            perpendicular = np.linalg.norm(delta - along[:, None] * light_direction, axis=1)
            candidates = np.where((along > 0) & (perpendicular < (radii + radius) * scale + along * .08))[0]
            casters = [(positions[j], radii[j] * scale) for j in candidates if j != index]
        canvas.sphere(
            position,
            radius * scale,
            _rgb(_element_color(element, scientific)),
            casters,
            opacity=settings.atom_opacity,
        )
    array = np.empty((h, w, 4), np.uint8)
    array[..., :3] = np.clip(canvas.rgb * 255, 0, 255).astype(np.uint8)
    array[..., 3] = np.clip(canvas.alpha * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(array)
    draw = ImageDraw.Draw(result)
    ink = "#23394b" if light else "#e4edf5"
    muted = "#627484" if light else "#9fb4c8"
    if settings.labels:
        occupied_rects = []
        for index in np.argsort(positions[:, 2])[::-1]:
            px, py, pz = positions[index]
            ix, iy = int(px), int(py)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            radius = radii[index] * scale
            if canvas.depth[iy, ix] > pz + radius + 1:
                continue
            text = scene.elements[index]
            font = _font(min(w * .017, max(10 * factor, radius * .63)), True)
            box = draw.textbbox((px, py), text, font=font, anchor="mm")
            if any(box[0] < q[2] and box[2] > q[0] and box[1] < q[3] and box[3] > q[1] for q in occupied_rects):
                continue
            occupied_rects.append(box)
            draw.text((px, py), text, font=font, fill="#142838", anchor="mm", stroke_width=0)
    if settings.axes:
        base = np.array([w * (.105 if scientific else .09), h * (.82 if scientific else (.79 if settings.caption else .86))])
        axis_length = min(w, h) * (.082 if scientific else .065)
        axis_colors = ("#e11820", "#12aa32", "#1630bd") if scientific else ("#d77573", "#52b59e", "#709bda")
        for vector, label, color in zip(scene.basis, "abc", axis_colors):
            v = rotation @ (vector / np.linalg.norm(vector))
            end = base + v[:2] * [axis_length, -axis_length]
            draw.line([tuple(base), tuple(end)], fill=color, width=max(2, int(w * .0025)))
            direction = end - base
            norm = np.linalg.norm(direction)
            if norm > 2:
                d = direction / norm
                side = np.array([-d[1], d[0]])
                back = end - d * w * .008
                draw.polygon([tuple(end), tuple(back + side * w * .0035), tuple(back - side * w * .0035)], fill=color)
                text_point = end + d * w * .011
            else:
                text_point = end + [0, -w * .016]
                draw.ellipse((end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=color)
            draw.text(tuple(text_point), label, font=_font(w * .017, True), fill=color, anchor="mm")
    if settings.caption and scientific:
        legend_elements = sorted(set(scene.elements), key=lambda el: ATOMIC_NUMBER.get(el, 999))
        count = max(1, len(legend_elements))
        step = min(h * .105, h * .56 / count)
        start_y = max(h * .10, h * .40 - step * (count - 1) / 2)
        for row, element in enumerate(legend_elements):
            cy = start_y + row * step
            cx = w * .13
            radius = min(w, h) * .027
            base_color = ImageColor.getrgb(_element_color(element, True))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=base_color, outline="#737b80", width=max(1, factor))
            highlight = radius * .30
            draw.ellipse((cx - radius * .42 - highlight, cy - radius * .42 - highlight,
                          cx - radius * .42 + highlight, cy - radius * .42 + highlight), fill="#fff8f4")
            draw.text((w * .045, cy), element, font=_font(w * .019, True), fill="#111111", anchor="lm")
    elif settings.caption:
        margin = w * .045
        baseline = h * .865
        draw.line((margin, baseline, w - margin, baseline), fill="#d6dee4" if light else "#3c5064", width=max(1, factor))
        title = str(scene.model.get("data_name") or "Crystal structure")
        title_font = _font(w * .023, True)
        # Bound arbitrary CIF names to the available title width.
        while len(title) > 1 and draw.textlength(title, font=title_font) > w * .54:
            title = title[:-2] + "…"
        draw.text((margin, h * .885), title, font=title_font, fill=ink)
        cell = scene.model["cell"]
        detail = "  ·  ".join(f"{key} {cell[key]:.4f}" for key in ("a", "b", "c")) + " Å"
        draw.text((margin, h * .926), detail, font=_font(w * .0125), fill=muted)
        is_result = scene.model.get("provenance", {}).get("kind") == "rietveld_result"
        note = "Rietveld cell · fixed CIF sites" if is_result else "CIF reference · not a refinement result"
        draw.text((margin, h * .954), note, font=_font(w * .0115), fill=muted)
        # Wrap the element key in a reserved right-hand column.
        x, y = w * .64, h * .893
        for element in sorted(set(scene.elements), key=lambda el: ATOMIC_NUMBER.get(el, 999)):
            if x + w * .09 > w * .97:
                x, y = w * .64, y + h * .034
            if y > h * .97:
                break
            r = w * .005
            draw.ellipse((x, y + r, x + 2 * r, y + 3 * r), fill=_element_color(element, False))
            draw.text((x + w * .017, y), element, font=_font(w * .014, True), fill=ink)
            x += w * .075
    provenance = scene.model.get("provenance") or {}
    if provenance.get("publication_ready") is False:
        status = str(provenance.get("validation_status") or "Structure review required")
        score = provenance.get("validation_score")
        score_text = "" if score is None else f" · score {float(score):.1f}"
        warning_text = f"STRUCTURE REVIEW REQUIRED{score_text} · {status}"
        font = _font(w * .013, True)
        box = draw.textbbox((0, 0), warning_text, font=font)
        pad_x, pad_y = w * .012, h * .008
        width_text = box[2] - box[0]
        x1, y0 = w * .965, h * .025
        x0, y1 = max(w * .24, x1 - width_text - 2 * pad_x), y0 + (box[3] - box[1]) + 2 * pad_y
        draw.rounded_rectangle((x0, y0, x1, y1), radius=max(3, int(h * .009)),
                               fill="#fff2dd", outline="#c56200", width=max(1, 2 * factor))
        draw.text(((x0 + x1) / 2, (y0 + y1) / 2), warning_text,
                  font=font, fill="#8a3400", anchor="mm")
    if factor > 1:
        result = result.resize((width, height), Image.Resampling.LANCZOS)
    return result


def export_crystal(scene: CrystalScene, path: str | Path, width=3200, height=2400, dpi=600) -> tuple[Path, Path]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".png", ".tif", ".tiff"}:
        raise ValueError("Export as PNG or TIFF to preserve image quality and transparency.")
    if not 72 <= dpi <= 1200:
        raise ValueError("DPI must be between 72 and 1,200.")
    picture = render_crystal(scene, width, height)
    document = scene_document(scene, width, height, dpi)
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    scene_path = path.with_suffix(".crystal.json")
    # Render fully before touching an existing figure.  Put the temporary files
    # directly in the destination directory: Python's private TemporaryDirectory
    # ACL can otherwise survive os.replace on Windows and make the exported PNG
    # unreadable to the interactive desktop user.
    token = uuid4().hex
    temp_image = path.with_name(f".{path.stem}.{token}{path.suffix}")
    temp_scene = scene_path.with_name(f".{scene_path.stem}.{token}{scene_path.suffix}")
    try:
        if suffix == ".png":
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("Description", document["conventions"])
            metadata.add_text("AfruzCrystalScene", serialized)
            picture.save(temp_image, dpi=(dpi, dpi), pnginfo=metadata)
        else:
            picture.save(temp_image, dpi=(dpi, dpi), compression="tiff_lzw", tiffinfo={270: serialized})
        temp_scene.write_text(serialized, encoding="utf-8")
        temp_scene.replace(scene_path)
        temp_image.replace(path)
    finally:
        temp_image.unlink(missing_ok=True)
        temp_scene.unlink(missing_ok=True)
    return path, scene_path
