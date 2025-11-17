"""
Utility functions for color prediction and naming.
"""
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000, delta_e_cmc
from typing import Dict, Literal
import numpy as np
import colour

# colormath uses a deprecated Numpy function
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item()


CMFS = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy() # color-matching functions
ILLUM = colour.SDS_ILLUMINANTS['D65'].copy().align(CMFS.shape) 

COLORED_SPECIES = { # (wavelength, FWHM, epsilon)
    'CrO4-2': (370, 55, 5000),
    'HCrO4-': (350, 55, 2000),
    'Cr2O7-2': [(350, 65, 3000), (440, 100, 400)],
    'Cu+2': (800, 230, 20),
    'Cu(NH3)4+2': (600, 160, 20),
    'Ni+2': [(400, 95, 12), (610, 75, 6), (690, 75, 7)],
    'Ni(NH3)6+2': [(375, 65, 20), (575, 55, 15)],
    'Co+2': [(466, 45, 6), (512, 62, 10), (600, 150, 0.7)],
    'Co(NH3)6+2': [(450, 45, 6), (500, 60, 9), (650, 140, 3), (550, 1000, 1.5)], # The last "peak" is there just to uniformly reduce the absorbance over all wavelengths, resulting in a "brownish" color
    'Fe2OH2+4': (340, 70, 4000),
    'FeCl+2': (335, 70, 1000), # DOI: 10.1016/j.chemgeo.2006.02.005
    'FeCl2+': (345, 70, 3800), # DOI: 10.1016/j.chemgeo.2006.02.005
    'FeCl3': (375, 80, 3000), # DOI: 10.1016/j.chemgeo.2006.02.005
    'FeSCN+2': (460, 85 , 5000),
    'Fe(SCN)2+': (485, 85, 9000),
    'FeOx+':     (300, 110, 1000),  # DOI: 10.1021/jp8040583
    'Fe(Ox)2-':  (300, 130, 2000),  # DOI: 10.1021/jp8040583
    'Fe(Ox)3-3': (300, 150, 3000),  # DOI: 10.1021/jp8040583
}


# Creating approximate molar absorptivity spectra of the above species using gaussian peaks.
SPECTRA = {}
for sp, peaks in COLORED_SPECIES.items():
    spectrum = colour.sd_zeros()
    if type(peaks) == tuple:
        peaks = [peaks]
    for peak in peaks:
        wavelength = peak[0]
        fwhm = peak[1]
        epsilon = peak[2]
        spectrum += colour.sd_gaussian(wavelength, fwhm, method='FWHM') * epsilon
    SPECTRA[sp] = spectrum


PRECIPITATE_COLORS = {# Many colors from Perry "Handbook of Inorganic Compounds"             
    'AgBr':      'pale yellow',         
    'Ag2CO3':    'pale yellow',            
    'Ag2CrO4':   'brick red',            
    'AgI' :      'yellow',                
    'Ag2O':      'dark brown',           
    'Ag3PO4':    'dirty yellow',               
    'Ag2S':      'black',                

    'BaCrO4':    'pale yellow',         

    'CaCrO4':    'yellow',          

    'CdCrO4':    'yellow',                
    'CdS':       'yellowish orange',     
    
    'CoCO3':     'pink',
    'Co(OH)2':   'blue',
    'CoOx':      'pink', # there is a photo in DOI: 10.1021/acssuschemeng.5b01000
    'Co3(PO4)2': 'lavender',


    'CuCrO4':    'reddish brown',        
    'CuCO3':     'turqoise',            
    'Cu(OH)2':   'blue',                
    'Cu3(PO4)2': 'turqoise',            
    'CuS':       'black',                 
    'CuOx':      'cyan',                  

    'Fe(OH)3':   'reddish brown',
    'Fe(OH)2':   'pale olive',
    'FeS':       'black',
    'Fe2S3':     'black',
    'FePO4':     'dirty yellow',
    'FeOx':      'pale yellow',

    'Hg':        'black',
    'HgCO3':     'brown',
    'Hg2CO3':    'yellow',
    'Hg2CrO4':   'brick red',
    'HgO':       'yellowish orange',
    'HgI2':      'orange-red',
    'Hg2I2':     'yellow',
    'Hg2F2':     'pale yellow',
    'HgS':       'black',
    'Hg3(PO4)2': 'pale yellow',

    'MnS':       'salmon',
    'Mn(OH)2':   'pale pink',
    
    'NiCO3':     'pale green',
    'NiCrO4':    'maroon', # this is weird but it was in Perry!!
    'Ni(OH)2':   'pale green',
    'Ni(Hdmg)2': 'crimson',
    'Ni(CN)2':   'pale turqoise',
    'NiOx':      'pale green',
    'Ni3(PO4)2': 'pale green',
    'NiS':       'black',

    'PbCrO4':    'yellow',
    'PbI2':      'yellow',
    'PbS':       'black',

    'SnS':       'black',

    'Tl2CrO4':   'yellow', # CRC
    'TlI':       'yellow',
    'Tl2S':      'black',

    'ZnCrO4':    'yellow'
}

PALLETT = {

    'salmon':           '#FA8062',
    'tomato':           '#FF6344',
    'orange-red':       '#EF5500',
    'red':              '#EE1111',
    'crimson':          '#D11120',
    'brick red':        '#A82B22',
    'dark red':         '#8B0000',
    'maroon':           '#601800',

    'coral':            '#E0685F',
    'dark orange':      '#D06000',
    'orange':           '#F07711',
    'yellowish orange': '#FFA822',
    'yellow':           '#FFDD00',
    'pale yellow':      '#FAF8AA',
    'dirty yellow':     '#D8D080',
    'wheat':            '#D5C5A0',
    'tan':              '#C2A48C',
    'rosy brown':       '#B08F8A',
    'light brown':      '#B19266',
    'brown':            '#774410',
    'reddish brown':    '#60302F',
    'dark brown':       '#382818',

    'greenish yellow':  '#CFE500',
    'yellowish green':  '#AAEE00',
    'lime':             '#7EEE00',
    'pale olive':       '#9BA970',
    'olive':            '#859535',
    'dark olive':       '#555D20',

    'pale turquoise':   '#AAFFCC',
    'pale green':       '#95EE95',
    'spring green':     '#00EE6E',
    'turquoise':        '#30CCAA',
    'green':            '#119922', 
    'teal-green':       '#008F5C',
    'teal':             '#008075',
    'dark green':       '#006400',
    'dark teal':        '#225050',

    'pale blue':        '#B0E0E6',
    'sky blue':         '#77CEEE',
    'cyan':             '#00E0E0',
    'dark cyan':        '#0090AA',
    'blue':             '#0055EE',
    'blue-gray':        '#445579',
    'dark blue':        '#0022BB',
    'navy blue':        '#000080',

    'dark indigo':      '#400099',
    'indigo-violet':    '#6E11EE',
    'deep purple':      '#800080',
    'dark violet':      '#8400C3',
    'magenta':          '#CC33CC',
    'pink':             '#EF83EF',
    'lavender':         '#BBAAFF',
    'thistle':          '#D8BFD8',
    'pale pink':        '#FFA0DD',
    'reddish gray':     '#796070',

    'white':            '#EEEEEE',
    'pale gray':        '#AAAAAA',
    'gray':             '#585858',
    'dark gray':        '#383838',
    'black':            '#111111',
}


# Binning and mapping HSL lightness values
L_BINS = np.array([0.20, 0.30, 0.45, 0.65, 0.75, 0.85, 0.91])
L_PREFIX = {
    1: 'dark',      # [0.20, 0.30)
    2: 'deep',      # [0.30, 0.45)
    3: '',          # [0.45, 0.65)
    4: 'light',     # [0.65, 0.75)
    5: 'pale',      # [0.75, 0.85)
    6: 'very pale', # [0.85, 0.91)
}

# Binning and mapping HSL hue values
H_BINS = np.array([8, 20, 33, 43, 67, 83, 105, 145, 157, 170, 190, 210, 235, 255, 270, 280, 295, 323, 340])
HUES = {
     0: 'red',                  # [  0,  8 )
     1: 'red-orange',           # [  8,  20)
     2: 'orange',               # [ 20,  33)
     3: 'yellow-orange',        # [ 33,  43)
     4: 'yellow',               # [ 43,  67)
     5: 'greenish yellow',      # [ 67,  83)
     6: 'yellowish green',      # [ 83, 105)
     7: 'green',                # [105, 145)
     8: 'green-turquoise',      # [145, 157)
     9: 'turquoise',            # [157, 170)
    10: 'cyan',                 # [170, 190)
    11: 'sky-blue',             # [190, 210)
    12: 'blue',                 # [210, 235)
    13: 'blue-violet',          # [235, 255)
    14: 'purple',               # [255, 270)
    15: 'purple-magenta',       # [270, 280)
    16: 'magenta',              # [280, 295)
    17: 'pink',                 # [295, 323)
    18: 'pinkish red',          # [323, 340)
    19: 'red'                   # [340, 360]
}

# creating the augmented pallet by adding slightly darker and lighter version of the colors 
AUG_PALLETT = {}
for name, hex in PALLETT.items():
    sRGB = sRGBColor.new_from_rgb_hex(hex)
    h, s, l = convert_color(sRGB, HSLColor).get_value_tuple()
    if 0.05 < l < 0.95:
        darker_hsl  = HSLColor(h, s, l-0.025)
        lighter_hsl = HSLColor(h, s, l+0.025)

        darker_hex  = convert_color(darker_hsl , sRGBColor).get_rgb_hex().upper()
        lighter_hex = convert_color(lighter_hsl, sRGBColor).get_rgb_hex().upper()

        aug = {
            lighter_hex: name,
            hex: name,
            darker_hex: name,
        }
    else:
        aug = {hex: name}

    AUG_PALLETT.update(aug)


def _hex_to_lab(hex: str) -> LabColor:
    srgb = sRGBColor.new_from_rgb_hex(hex)
    lab = convert_color(srgb, LabColor, target_illuminant='d65')
    return lab


def _hex_to_linRGB(hex: str) -> tuple[float]:
    sRGB = sRGBColor.new_from_rgb_hex(hex).get_value_tuple()
    s2lin = lambda x: x/12.92 if x<= 0.04045 else ((x + 0.055)/1.055)**2.4
    linRGB = tuple([s2lin(c) for c in sRGB])
    return linRGB


def _linRGB_to_hex(linRGB: tuple[float]) -> str:
    lin2s = lambda x: 12.92*x if x <= 0.0031308 else 1.055*(x ** (1/2.4)) - 0.055
    r, g, b = [round(255*lin2s(c)) for c in linRGB]
    return f"#{r:02X}{g:02X}{b:02X}"


def mix_colors(mixture: list[tuple[str, float]], blend: float=0.15) -> str:
    """
    This function is used to predict precipitate color.
    It mixes the given hex colors in linear RGB, using a convex combination of
    geometric (proxy for subtractive) and arithmetic (additive) means.

    Each input color is converted from hex to linear RGB, mixed according to its
    fraction, and the result is blended between arithmetic and geometric mixing.
    A small floor (0.01) is applied per-channel before the geometric step to avoid
    collapse to zero. The final linear RGB triplet is converted back to a hex color.

    Parameters
    ----------
    mixture : list[tuple[str, float]]
        Sequence of (hex_color, fraction) pairs. Hex must be in the form "#RRGGBB"
        or "RRGGBB". Fractions must be positive and sum to 1.0.
    blend : float, default=0.15
        Interpolation weight between arithmetic and geometric mixes in linear RGB:
            result = blend · A + (1 - blend) · G
        Use 1.0 for purely additive (arithmetic) mixing, 0.0 for purely
        subtractive-like (geometric) mixing. Must be within [0, 1].

    Returns
    -------
    str
        The resulting color as a hex string "#RRGGBB".

    Raises
    ------
    AssertionError
        If any fraction is non-positive or if the fractions do not sum to 1.0
        when rounded to 3 decimal places.
    ValueError
        If blend is outside the 0-1 range. 
    """
    fracs = [fraction for _,fraction in mixture]

    assert (round(sum(fracs),3) == 1) and all(frac > 0 for frac in fracs), "All fractions must be positive and sum to one!"
    if (blend < 0) or (blend > 1):
        raise ValueError("blend must be within [0, 1]")

    arithmetic = [0, 0, 0]
    geometric = [1, 1, 1]

    for hex, frac in mixture:
        r, g, b = _hex_to_linRGB(hex)
        # setting the min to avoid collapse in geometric (subtractive) mode
        r = max(0.01, r)
        g = max(0.01, g)
        b = max(0.01, b)

        arithmetic[0] += r * frac
        arithmetic[1] += g * frac
        arithmetic[2] += b * frac

        geometric[0] *= r ** frac
        geometric[1] *= g ** frac
        geometric[2] *= b ** frac

    blended = tuple([blend * arithmetic[i] + (1-blend) * geometric[i] for i in range(3)])
    return _linRGB_to_hex(blended)


def absorption_spectrum(composition: Dict[str, float]) -> colour.SpectralDistribution:
    """
    Uses Beer-Lambert additivity to calculate the approximate absorption
    spectrum of a solution. 
    The simplified species spectra from `SPECTRA` are used. 
    """

    sol_spec = colour.sd_zeros(name='absorbance spectrum')
    for sp, spectrum in SPECTRA.items():
        conc = composition.get(sp, 0)
        sol_spec += spectrum * conc
    return sol_spec


def solution_color(composition: Dict[str, float], optical_path_length: float=2, out: Literal['rgb', 'hex']='hex') -> str | tuple[float]:
    """
    Predict the perceived color of a solution from its composition using
    Beer–Lambert law and colorimetric rendering to sRGB.

    First, the absorbance spectrum is calculated from `composition`
    and is scaled by `optical_path_length`. Then absorbance is converted
    to transmittance: T(λ) = 10^-A(λ).
    Transmittance specturm is converted to tristimulus values using 'D65'
    as illuminant and 'CIE 1931 2 Degree Standard Observer' as the color-
    matchin functions. XYZ color is then converted to sRGB or hex.

    Parameters
    ----------
    composition : Dict[str, float]
        species → concentration (molar)
    optical_path_length : float, default=2
        Optical path length multiplier
    out : {'rgb', 'hex'}, default='hex'
        Output format: sRGB triplet in [0,1] (`'rgb'`) or hex string (`'hex'`).

    Returns
    -------
    str | tuple[float]
        - If `out='hex'`: hex color string "#RRGGBB".
        - If `out='rgb'`: tuple of sRGB components in [0, 1].

    Raises
    ------
    ValueError
        If `out` is not `'rgb'` or `'hex'`.
    """

    abs_spec = absorption_spectrum(composition)
    abs = abs_spec * optical_path_length
    transmittance = (colour.sd_constant(0.1).copy() ** (abs)).align(CMFS.shape)
    transmittance.normalise()

    xyz = colour.sd_to_XYZ(transmittance, CMFS, ILLUM)
    rgb = colour.XYZ_to_sRGB(xyz/100)
    rgb = np.clip(rgb, 0, 1)
    if out.lower() == 'rgb':
        return tuple(rgb)
    elif out.lower() == 'hex':
        hex = sRGBColor(*rgb).get_rgb_hex()
        return hex
    else:
        raise ValueError(f"Invalid output type: {out!r}. Must be either 'rgb' or 'hex'. ")
    

def closest_color_names(target_hex: str, mode: Literal['precipitate', 'solution'], max_names: int=3, debug: bool=False) -> str:
    """
    Returns a human-readable color name (or names) from a target hex color.

    Two naming strategies are supported:

    1) Precipitate mode
       - Interprets `target_hex` as a solid/opaque color.
       - Converts to CIE Lab (D65) and compares against `AUG_PALLETT` using both CMC and CIE2000 ΔE.
       - The closest match from each metric is kept if CIE ΔE < 17 or CMC ΔE < 18. Additionally,
         any other candidates (up to six) within 3 ΔE (for CIE) or 2 ΔE (for CMC) of that best score are included.
       - The candidates are then sorted by ΔE and at most, the top `max_names` different color names are reported. 
       - If no candidate passes the ΔE gate in either metric, returns `'muddy'`.

    2) Solution mode
       - Interprets `target_hex` as a translucent/solution-like color.
       - Converts to HSL.
       - First classifies lightness `L`:
           * L ≤ 0.2  -> "almost black"
           * L ≥ 0.91  -> "colorless"
           * S < 0.6 and L < 0.6  -> "muddy"
           * otherwise chooses a prefix from `L_PREFIX` by binning L with `L_BINS`
       - Then classifies hue `H` by binning with `H_BINS` and mapping to `HUES`.
       - Brown is treated as a special case (dark or low saturation orange/yellow-orange)
       - Returns either the bare hue (when prefix is empty) or "<prefix> <hue>".

    Parameters
    ----------
    target_hex : str
        Color in hex notation "#RRGGBB" or "RRGGBB".
    mode : Literal['precipitate', 'solution']
        Naming strategy. See descriptions above.
    max_names : int
        Maximum number of color names to keep in precipitate mode.
    debug : bool
        Whether or not to print additional debug information

    Returns
    -------
    str
        - In 'precipitate' mode: candidate names separated by slashes, or 'muddy' if no close match.
        - In 'solution' mode: a single descriptive name string.

    Raises
    ------
    ValueError
        If `target_hex` is not a valid hex string, or if `mode` is not one of `'precipitate'` or `'solution'`.

    """
    try:
        target_rgb = sRGBColor.new_from_rgb_hex(target_hex)
    except ValueError:
        raise ValueError("Invalid hex color format! Must be either 'RRGGBB' or '#RRGGBB'.")

    if mode == 'precipitate':
    # Precipitate Mode: target is converted to LabColor and compared to colors in AUG_PALLETT
    #                   If no close match is found, returns 'muddy'
    #                   If multiple close matches are found, all are reported as a set

        target_lab = convert_color(target_rgb, LabColor, target_illuminant='d65')

        cmc = []
        cie = []
        candidates = []

        for hex in AUG_PALLETT:
            lab = _hex_to_lab(hex)
            cmc.append({'hex': hex, 'delta': delta_e_cmc(target_lab, lab, pl=1)})
            cie.append({'hex': hex, 'delta': delta_e_cie2000(target_lab, lab)})

        by_delta = lambda d: d['delta']
        cmc.sort(key=by_delta)
        cie.sort(key=by_delta)

        if cmc[0]['delta'] < 18:
            candidates.append(cmc[0])
            for i in range(1,10):
                diff = cmc[i]['delta'] - cmc[0]['delta']
                if diff <= 2:
                    candidates.append(cmc[i])

        if cie[0]['delta'] < 17:
            candidates.append(cie[0])
            for i in range(1,10):
                diff = cie[i]['delta'] - cie[0]['delta']
                if diff <= 3:
                    candidates.append(cie[i])
        if debug:
            print("---> Debug Info:")
            print("     CIE: ", cie[:6])
            print("     CMC: ", cmc[:6])
            print("-----  ")
            
        candidates.sort(key=by_delta)
        names = []
        for candidate in candidates:
            name = AUG_PALLETT[candidate['hex']]
            if name not in names:
                names.append(name)

        if len(names) > 0:
            return ' / '.join(names[:max_names])
        else:
            return 'muddy'
    
    elif mode == 'solution':
    # Solution Mode: Color is converted to HSL, L is evaluated first.
    #                If L is not in the 0.2-0.91 range, hue is not checked and either 'almost black' or 'colorless' are returned.
    #                If L is in the 0.2-0.91 range, and both S and L are less than 0.6, prefix is 'muddy',
    #                otherwise the prefix is selected based on the value of L.
    #                Then hue is selected based on the value of H.
    #                Brown is treated as a special case of orange/yellow-orange
    #                Finally, prefix + hue is returned.

        h, s, l = convert_color(target_rgb, HSLColor).get_value_tuple()

        if debug:
            print("---> Debug Info:")
            print(f"     {h=:.2f}  {s=:.2f}  {l=:.2f}")
            print("-----  ")

        if l <= 0.2 :
            return 'almost black'
        elif l >= 0.91 :
            return 'colorless'
        elif (s < 0.6) and (l < 0.6):
            prefix = 'muddy'
        else:
            prefix = L_PREFIX[np.digitize(l, L_BINS)]
        
        hue = HUES[np.digitize(h, H_BINS)]

        # special case of (light) brown
        if (hue in ['orange', 'yellow-orange']) and (prefix in ['muddy', 'deep', 'dark']):
                return 'brown'
        elif (hue in ['orange', 'yellow-orange']) and (s <= 0.5) and (l >= 0.5):
                return 'light brown'
        # not brown
        elif prefix == '':
            return hue
        else:
            return prefix + ' ' + hue
    
    # wrong mode
    else:
        raise ValueError("Invalid mode! Can only be 'precipitate' or 'solution'")
