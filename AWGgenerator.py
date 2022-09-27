import numpy as np
import os
from math import *

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def str2(var):
    return var if isinstance(var, str) else str(var)

def read_txt(fname, startLine = 1, sep = '\t'):
    with open(fname, 'r') as lines:
        for i, line in enumerate(lines):
            if i == startLine:
                data = np.array([float(num) for num in line.rstrip('\n').replace(' ', '\t').split(sep)])
            if i > startLine:
                data = np.insert(np.array([[float(num) for num in line.rstrip('\n').replace(' ', '\t').split(sep)]]), 0, values = data, axis = 0) if line != '\n' else data     # attention!!! the '\n' at the EOF should be eliminated (if line != '\n' else data)
    return data


def nslab_fun(wl0, tg, nfFName, nsubFName, ncFName, pol = 0):
    DEte = lambda b, k0, tg, m, nf, ns, nc: k0* tg * np.sqrt(nf**2-ns**2) *np.sqrt(1-b) - np.arctan(np.sqrt((b+(ns**2-nc**2)/(nf**2-ns**2))/(1-b))) - np.arctan(np.sqrt((b)/(1-b))) - m*pi
    DEtm = lambda b, k0, tg, m, nf, ns, nc: k0* tg * np.sqrt(nf**2-ns**2) * np.sqrt(1-b) - np.arctan(nf**2 / nc**2 * ((b+(ns**2-nc**2)/(nf**2-ns**2))/np.sqrt(1-b))) - np.arctan(nf**2 / ns**2 * np.sqrt((b)/(1-b))) - m*pi
    # h_fun = lambda b, k0, nf, ns: k0* np.sqrt(nf**2-ns**2) *np.sqrt(1-b)
    # p_fun = lambda b, k0, nf, ns, nc: h_fun(b, k0, nf, ns) * np.sqrt((b)/(1-b))
    # q_fun = lambda b, k0, nf, ns, nc: h_fun(b, k0, nf, ns) * np.sqrt((b+(ns**2-nc**2)/(nf**2-ns**2))/(1-b))
    neff_fun = lambda brt, nf, ns: sqrt(brt*(nf**2-ns**2)+ns**2)
    # EyN_fun0 = lambda x, h, p, q, k0, tg, nf, ns, nc: np.cos(h*x)-q/h*np.sin(h*x) if (x >= -tg and x < 0) else ((np.cos(h*tg) + q/h*np.sin(h*tg))*np.exp(p*(x+tg)) if x < -tg else np.exp(-q*x))
    # EyN_fun1 = lambda x, h, p, q, k0, tg, nf, ns, nc: EyN_fun0(x-tg/2, h, p, q, k0, tg, nf, ns, nc) / EyN_fun0(-tg/2, h, p, q, k0, tg, nf, ns, nc)
    # EyN_fun = np.vectorize(EyN_fun1)
    
    k0 = 2*pi / wl0;
    
    nf_data = read_txt(nfFName, startLine = 0)
    nf_fun = np.vectorize(    lambda wl: np.interp(wl, nf_data[:, 0], nf_data[:, 1])    )
    nf = nf_fun(wl0)
    
    ns_data = read_txt(nsubFName, startLine = 0)
    ns_fun = np.vectorize(    lambda wl: np.interp(wl, ns_data[:, 0], ns_data[:, 1])    )
    ns = ns_fun(wl0)
    
    nc_data = read_txt(ncFName, startLine = 0)
    nc_fun = np.vectorize(    lambda wl: np.interp(wl, nc_data[:, 0], nc_data[:, 1])    )
    nc = nc_fun(wl0)
    
    b = np.linspace(0, 0.999999, 6001)
    de = DEte(b, k0, tg, 0, nf, ns, nc) if pol == 0 else DEtm(b, k0, tg, 0, nf, ns, nc)
    brt = np.interp(0, de[::-1], b[::-1])
    # h = h_fun(brt, k0, nf, ns)
    # p = p_fun(brt, k0, nf, ns, nc)
    # q = q_fun(brt, k0, nf, ns, nc)
    nslab = neff_fun(brt, nf, ns)
    print('nslab = ', nslab)
    return nslab


def exportRaFile(naFName, nfFName, nsubFName, ncFName, narFName, Lambda0, DLambda, Di, Do, Nchan, M, Md, Wot, Lot, Width, arcend_pitch, ns_height = 0.22, nslabFName = None, overwrite = 0):
    na_data = read_txt(naFName, startLine = 0)
    na = np.vectorize(    lambda wl: np.interp(wl, na_data[:, 0], na_data[:, 1])    )  # red shift to calibrate the expriment data
    ng = np.vectorize(    lambda wl: na(wl) - wl*(na(wl+0.0001)-na(wl-0.0001))/(0.0002)    )  # red shift to calibrate the expriment data
    
    if nslabFName != None:
        ns_data = read_txt(nslabFName, startLine = 0)
        ns_fun = np.vectorize(    lambda wl: np.interp(wl, ns_data[:, 0], ns_data[:, 1])    )  # red shift to calibrate the expriment data
        ns = ns_fun(Lambda0)
    if nfFName != None and nsubFName != None and ncFName != None:
        ns = nslab_fun(wl0 = Lambda0, tg = ns_height, nfFName = nfFName, nsubFName = nsubFName, ncFName = ncFName, pol = 0)
    
    FSR = Nchan * DLambda
    mp = Lambda0 / FSR
    DispersionFactor = ng(Lambda0) / na(Lambda0)
    m = round(mp / DispersionFactor)
    Ro = ns*Di*Do / (m*DispersionFactor*DLambda)
    print('Ro = ', Ro)
    Ao = Do/Ro
    Asa = (M+2*Md+1)/2
    Nslab_design = ns
    W = Width
    print('Lot = ', Lot)
    Lot = eval(Lot)
    print('Lot = ', Lot)
    Lo = 2*Lot
    print('Lo = ', Lo)
    idxRa = np.linspace(1, M, M)
    zAai = (idxRa-Asa)*Ao
    Rai = ((idxRa-Asa)*arcend_pitch - (Ro+Lo)*np.sin(zAai)) / (1-np.cos(zAai))
    numRai = round(np.size(Rai, 0)/2)
    Rai_data = np.transpose(np.vstack([abs(Rai[0:numRai]), np.ones(numRai)]))
    print('Rai = ', Rai)
    if overwrite == 1:
        np.savetxt(narFName, Rai_data, delimiter = ' ')


def writeBaiscParam(fileName,               # target .ind filename
                                    Nin,                        # number of the input port of the AWG
                                    Nout,                      # number of the output port of the AWG
                                    Nchan,                   # number of the channel, usually  = Nout + 2
                                    M,                          # number of the arrayed waveguide, should be enough to contain the 1/e of the Fourier diffreaction pattern of the input port
                                    Di,                          # waveguide spacing of the Rowland inner circle (input/output ports)
                                    Do,                         # waveguide spacing of the Rowland outer circle (arrayed waveguide)
                                    Lambda0,                # central wavelength in free space, in um
                                    DLambda,                # channel spacing, um
                                    Wit,                        # width of the taper on Rowland inner circle (input/output ports)
                                    Wot,                        # width of the taper on Rowland outer circle (arrayed waveguides)
                                    Width,                     # width of the strip-loded waveguide (or rib/ridge waveguide)
                                    Lit = '(Nslab_design/Lambda0)*(Wit*Wit)',           # the expression should be allowable in both rsoft cad and python
                                    Lot = '(Nslab_design/Lambda0)*(Wot*Wot)',       # the expression should be allowable in both rsoft cad and python
                                    arcend_pitch = 2.6,                                                     # pitch of AWs at the slab interfaces    
                                    DL_pitch = 4,                                                               # pitch of AWs at the rest of incremental region
                                    arcend_straightBuffer = 0,                                          # to fine tune the straight region which is right after arc region, no effect on overall phase difference
                                    arc_radius = 10,                                                            # for 90-degree bending waveguide
                                    bendingOffset = 0,                                                  # to fine tune the waveguide offset between 90-degree and the straigth(tapered) waveguides
                                    # Lvertical_min = 10,                         # doesnt matter cause actual implemented DLi = DLi - DL1, DLi = i*DL - DLarc_for_pitch - DLstraight_for_constant_z
                                    Lh_min = 0.1,                                   # no effect on overall phase difference
                                    Width_fat = None,                                           # fat width to diminish phase errors due to fabricated side-wall roughness
                                    Nd = 0,                                                             # middle index shift of the input(output) port waveguides
                                    Md = 0,                                                             # middle index shift of the AWs
                                    naFName = 'na.txt',                                                         # effective index of the array waveguides(AWs) with the excess length DL (delta_L)
                                    narFName = 'naR.txt', narFileOverwrite = 0,             # to fine tune the length difference due to arc for a pitch
                                    nslabFName = 'ns.txt',                                          # slab effective index at star coupler, to calculate Ro, important issue, could be replaced with nf(ns/nc)FName followed by 1D dispersion-equation-root calculation for neffslab
                                    nfFName = 'Si_material.dat',                                # film material file, to calculate neffslab if narFName is not provided, important for Ro calculation
                                    nsubFName  = 'SiO2_material.dat',                       # substrate material file, to calculate neffslab if narFName is not provided, important for Ro calculation
                                    ncFName = 'SiO2_material.dat',                          # cladding material file, to calculate neffslab if narFName is not provided, important for Ro calculation
                                    height = 0.22,                                                      # height of the slab in star coupler region
                                    inputCenterx = 1000,                                        # to avoid the GDS-II vertices bug when exported from RSoft Ind file for pattern not in the first quadrant
                                    inputCenterz = 100,                                         # to avoid the GDS-II vertices bug when exported from RSoft Ind file for pattern not in the first quadrant
                                    nDL = 2,        # minimum amount of DL for DLi, can be considered as the minimum straight length for all AWs; doesnt matter since actual DLi = DLi - DL1
                                    port_pitch = 3,         # pitch of the input(output) port waveguides, unit in um
                                    Ltfat = 10,          # the length for transition for fat waveguides
                                    excess_width_cld = 4,       # trench width, unit in um
                                    shape = 2,                          # 0 for smit (not completed here); 1 for rectangular; 2 for S-shaped;
                                    ):
    basicParamStr = [
        'Ai = (Di/Ri)*(180/pi)',                        # pitch angle of input/output waveguides, inner circle, in degree
        'Ao = (Do/Ro)*(180/pi)',                    # pitch angle of input/output waveguides, outer circle, in degree
        'Asa = (M+2*Md+1)/2',                       # middle index of the arrayed waveguides (can be double)
        'Asi = (Nin+2*Nd+1)/2',                     # middle index of the input waveguides (can be double)
        'Aso = (Nout+2*Nd+1)/2',                    # middle index of the output wavegudies (can be double)
        'CAi = sinc(Ai/2)',                                 #  for smit version only
        'CAo = sinc(Ao/2)',                             # for smit version only
        'DL = GratingOrder*Lambda0/Nguide_design',          # delta_L, length difference of the adjacent waveguide
        'DL_pitch = ' + str(DL_pitch),                                  # pitch for actual incremental region
        'DLambda = ' + str(DLambda),                                                       # delta_Lambda, channel spacing
        'Design_background_index = nreal($background_material,Lambda0)',
        'Design_cover_index = nreal($cover_material,Lambda0)',
        'Design_delta = Design_slab_index-Design_background_index',
        'Design_height = height',
        'Design_pol = polarization',
        'Design_slab_height = slab_height',
        'Design_slab_index = nreal($slab_material,Lambda0)',
        'Design_width = width',
        'Di = ' + str(Di),                          # waveguide separation (along arc) of input/output ports
        'DispersionCorrection = 1',
        'DispersionFactor = if(DispersionCorrection,Ngroup_design/Nguide_design,1)',
        'Do = ' + str(Do),                     # waveguide separation (along arc) of the array waveguides
        'Dz = 0',                               # central difference between inner and outer circles
        'GratingOrder = round(Lambda0/(Nchan*DLambda*DispersionFactor))',           # grating order, the most important part of AWG
        'HasNaGF = defined("Nguide_file")',                                 # if na file exists
        'HasNsGF = defined("Neffslab_file")',                               # if neffslab file exists        
        'Hside = Design_slab_height',
        'Lambda0 = ' + str(Lambda0),
        'Li = 2*Lit',
        'Lic = Ri*(1-cos(Ai/2))',
        'Lim = max(Ro/100,Lic)',
        'Lit = ' + str2(Lit),
        'Litm = Lit-0.1',
        'Lo = 2*Lot',
        'Loc = Ro*(1-cos(Ao/2))',
        'Lom = max(Ro/100,Loc)',
        'Lot = ' + str2(Lot), 
        'Lotm = Lot-0.1',
        'Lsep = Ri+Dz',       # distance between the center of inner and the one of outer circle
        'Lstar = (Ri+Ro)-(Sagi+Sago)-Lsep',
        'M = ' + str(M),                    # > Ro * divergence_angle *2/ Do, divergance_angle = Lambda0 / pi/W0/ns
        'Md = ' + str(Md),
        'NC = Design_cover_index',
        'NF = Design_background_index+Design_delta',
        'NS = Design_background_index',
        'Nchan = ' + str(Nchan),
        'Nd = ' + str(Nd),
        'Ngroup2D = slabneff(Design_pol,0,Lambda0,NS,NS,NF,Design_width,1)',
        'Ngroup3D = eimneff(Design_pol,0,0,Lambda0,NC,NS,NF,Design_width,Design_height,Hside,1)',
        # 'NgroupNGF = NguideNGF-Lambda0*userderiv($Nguide_file:"",Lambda0)',
        'NgroupNGF = NguideNGF-Lambda0*(NguideNGF1-NguideNGF0)/0.0002',
        'Ngroup_design = if(HasNaGF,NgroupNGF,if(is2D,Ngroup2D,Ngroup3D))',
        'Nguide2D = slabneff(Design_pol,0,Lambda0,NS,NS,NF,Design_width)',
        'Nguide3D = eimneff(Design_pol,0,0,Lambda0,NC,NS,NF,Design_width,Design_height,Hside)',
        'NguideNGF = userdata($Nguide_file:"",Lambda0)',
        'NguideNGF0 = userdata($Nguide_file:"",Lambda0-0.0001)',
        'NguideNGF1 = userdata($Nguide_file:"",Lambda0+0.0001)',
        'Nguide_design = if(HasNaGF,NguideNGF,if(is2D,Nguide2D,Nguide3D))',
        'Nguide_file = ' + naFName,
        # 'Nguide_file_actual = ' + naFName,
        'NeffR_file = ' + narFName, 
        'Nin = ' + str(Nin),
        'Nout = ' + str(Nout),
        'Nslab2D = NF',
        'Nslab3D = slabneff(Design_pol,0,Lambda0,NC,NS,NF,Design_height)',
        'Nslab_design = if(HasNsGF,NeffslabNGF,if(is2D,Nslab2D,Nslab3D))',
        'NeffslabNGF = userdata($Neffslab_file:"",Lambda0)',
        'Neffslab_file = ' + nslabFName if nslabFName != None else '', 
        'Ri = Ro/2',
        'Ro = (Nslab_design/GratingOrder)*((Di*Do)/(DLambda*DispersionFactor))',
        'Sagi = Ri-sqrt(Ri^2-(Wstar/2)^2)',
        'Sago = Ro-sqrt(Ro^2-(Wstar/2)^2)',
        'W = width',
        'Wim = W+(Wit-W)*(Lit+Lim)/(Lit+Lic)',
        'Wit = ' + str(Wit),
        'Wom = W+(Wot-W)*(Lot+Lom)/(Lot+Loc)',
        'Wot = ' + str2(Wot),
        'Wstar = min(WstarMax,(M+2*Md+4)*Do)',
        'WstarMax = if(ge(Ro,Ri*sqrt(2)),2*Ri,2*Ro*sqrt(1-(0.5*Ro/Ri)^2))',
        'Zlens = Lsep+(Sagi-Ri)+Lstar/2',
        'alpha = 0',
        'background_alpha = nimag($background_material)',
        'background_index = nreal($background_material)',
        'background_material = SiO2',
        'cad_aspectratio_x = 1',
        'cad_aspectratio_y = 1',
        'char_delta = Design_delta',
        'char_height = Design_height',
        'char_width = Design_width',
        'cover_alpha = nimag($cover_material)',
        'cover_index = nreal($cover_material)',
        'cover_material = SiO2',
        'datapath = ..',
        'default_material = $slab_material',
        'delta = 0',
        'dimension = 3',        # doesnt matter, since the na file must be provided
        'domain_delta = width',
        'domain_max = inputCenterz+domain_bDL+(DL' + str2(M) + '-DL1)/2+arc_radius+2*Ltfat',
        'domain_bDL = arcend_straightBufferConstant+eq(shape,2)*(M-1)*DL_pitch', 
        'domain_min = inputCenterz-Li',
        'domain_round_sym = 1',
        'eim = 0',
        'free_space_wavelength = Lambda0',
        # 'grid_size = 0.025',
        # 'grid_size_y = 0.02',
        'height = ' + str(height),
        'is2D = eq(dimension,2)',
        'is2Deff = or(is2D,eim)',
        'k0 = (2*pi)/free_space_wavelength',
        # 'launch_align_file = 1',
        # 'launch_file = mode.m00',
        # 'launch_tilt = 1',
        # 'launch_type = if(is2Deff,LAUNCH_WGMODE,LAUNCH_FILE)',
        # 'launch_type_reset = 1',
        # 'modelist_output = 1',
        # 'monitor_file = mode.m00',
        # 'monitor_step_size = 5*step_size',
        # 'monitor_type = if(is2Deff,MONITOR_WGMODE_POWER,MONITOR_FILE_POWER)',
        # 'p0_file_out = power.dat',
        # 'pade_order = 0',
        # 'pathway_overlap_warning = 0',
        'polarization = 0',                                                                 # 0 for TE, 1 for TM
        # 'sim_tool = ST_BEAMPROP',
        'slab_alpha = nimag($slab_material)',
        'slab_height = 0',
        'slab_index = nreal($slab_material)',
        'slab_material = Core',                                                       # doesnt matter
        'slice_display_mode = DISPLAY_CONTOURMAPXZ',
        'slice_output_format = OUTPUT_NONE',
        # 'step_size = 0.125',
        'structure = STRUCT_RIBRIDGE',
        'vector = 2',
        'width = ' + str(Width),        # (port) waveguide width
        'width_2 = ' + str2(1.5*Width) if Width_fat == None else 'width_2 = ' + str2(Width_fat),
        'width_taper_in = TAPER_LINEAR',
        'width_taper_out = TAPER_LINEAR',
        'arcend_pitch = ' + str2(arcend_pitch), 
        'inputCenterx = ' + str2(inputCenterx),             # to avoid the bug of not being in the first quaddrant
        'inputCenterz = ' + str2(inputCenterz),             # to avoid the bug of not being in the first quadrant
        'arcend_straightBuffer = ' + str2(arcend_straightBuffer), 
        'arcend_straightBufferConstant = (Ro+Lo)*cos(0.5*Ao)+Ra' + str((M+2*Md+1+1)//2) + '*sin(0.5*Ao)+arcend_straightBuffer', 
        # 'Lvertical_min = ' + str2(Lvertical_min), 
        'Lh_min = ' + str2(Lh_min), 
        'arc_radius = ' + str2(arc_radius), 
        'nDL = ' + str2(nDL), 
        'bendingOffset = ' + str2(bendingOffset), 
        'port_pitch = ' + str2(port_pitch), 
        'excess_width_cld = ' + str2(excess_width_cld), 
        'shape = ' + str2(shape),       # 1 for box; 2 for S-shaped
        'outputSign = ' + str(1) if shape ==1 else 'outputSign = ' + str(-1), 
    ]
    
    ## degree of the arrayed region with the ref. @ middle one
    for i in range(M):
        basicParamStr.append('zAa' + str(i+1) + ' = (' + str(i+1) + '-Asa)*Ao')
    
    ## degree of  the input port
    for i in range(Nin):
        basicParamStr.append('zAi' + str(i+1) + ' = 180-(' + str(i+1) + '-Asi)*Ai')
    
    ## degree of the output port
    for i in range(Nout):
        basicParamStr.append('zAo' + str(i+1) + ' = 180-(' + str(i+1) + '-Aso)*Ai')
    
    ## arc radius for arrayed region
    for i in range(M):
        basicParamStr.append('Ra' + str(i+1) + ' = ((' + str(i+1) + '-Asa)*arcend_pitch-(Ro+Lo)*sin(zAa' + str(i+1) + '))/(1-cos(zAa' + str(i+1) + '))')
    
    ## arc radius for output port
    for i in range(Nout):
        basicParamStr.append('Ro' + str(i+1) + ' = ((' + str(i+1) + '-Aso)*port_pitch-(Ro+Li)*sin((180-zAo' + str(i+1) + ')/2))/(1-cos((180-zAo' + str(i+1) + ')/2))')
    
    ## arc radius for input port
    for i in range(Nin):
        if Nin%2 != 1 or i != Nin//2:
            basicParamStr.append('Ri' + str(i+1) + ' = ((' + str(i+1) + '-Asi)*port_pitch-(Ro+Li)*sin((180-zAi' + str(i+1) + ')/2))/(1-cos((180-zAi' + str(i+1) + ')/2))')
    
    ## arc straight buffer
    for i in range(M):
        basicParamStr.append('arcStraightBuffer' + str(i+1) + ' = arcend_straightBufferConstant-((Ro+Lo)*cos(zAa' + str(i+1) + ')+Ra' + str(i+1) + '*sin(zAa' + str(i+1) + '))')
    
    ## phase cor due to the arc
    for i in range(M):
        basicParamStr.append('DLradius' + str(i+1) + ' = -2*NeffRa' + str(i+1) + '/Nguide_design*Ra' + str(i+1) + '*zAa' + str(i+1) + '*pi/180')
    
    ## phase cor due to the arcend_pitch
    for i in range(M):
        basicParamStr.append('DLarcstraightBuffer' + str(i+1) + ' = -2*arcStraightBuffer' + str(i+1))
    
    ## DL after elimination of the effect of the arc and the arc straight buffer
    for i in range(M):
        basicParamStr.append('DL' + str(i+1) + ' = DL*(nDL+' + str(i+1) + ')+DLradius' + str(i+1) + '+DLarcstraightBuffer' + str(i+1) + '+-2*eq(shape,1)*arcend_pitch*' + str(i))
    
    ## NeffRai
    for i in range(M):
        basicParamStr.append('NeffRa' + str(i+1) + ' = userdata($NeffR_file:"", abs(Ra' + str(i+1) + '))')
    
    ## overlapped star couplers with different height
    basicParamStr.extend([
        'Ririb = Ri+Lirib',
        'Rorib = Ro+Lorib',
        'Riflr = Ri+Lit+0.5',
        'Roflr = Ro+Lot+0.5',
        'Sagirib = Ririb-sqrt(Ririb^2-(Wstarrib/2)^2)',
        'Sagiflr = Riflr-sqrt(Riflr^2-(Wstarflr/2)^2)',
        'Sagorib = Rorib-sqrt(Rorib^2-(Wstarrib/2)^2)',
        'Sagoflr = Roflr-sqrt(Roflr^2-(Wstarflr/2)^2)',
        'Ltfat = ' + str2(Ltfat), 
        'Lseprib = Lsep',
        'Lsepflr = Lsep',
        'Lirib = Lit/2+0.5',
        'Lorib = Lot/2+0.5',
        'Lstarrib = (Ririb+Rorib)-(Sagirib+Sagorib)-Lseprib',
        'Lstarflr = (Riflr+Roflr)-(Sagiflr+Sagoflr)-Lsepflr',
        'Zlensrib = (Sagirib-Ririb)+Lstarrib/2+Lseprib',
        'Zlensflr = (Sagiflr-Riflr)+Lstarflr/2+Lsepflr',
        'Hrib = 0.15',      # 0.15
        'Wstarrib = Wstar*(1+Lo/Ro)',
        'Wstarflr = Wstarrib+2',
    ])
    with open(fileName, 'w') as foh:
        foh.write('\n'.join(basicParamStr))
        foh.write('\n\n\n')
      
    exportRaFile(naFName = naFName, narFName = narFName, nslabFName = nslabFName, 
            nfFName = nfFName, nsubFName = nsubFName, ncFName = ncFName, ns_height = height, Lambda0 = Lambda0, 
            DLambda = DLambda, Di = Di, Do = Do, Nchan = Nchan, M = M, Md = Md, 
            Wot = Wot, Lot = Lot, Width = Width, arcend_pitch = arcend_pitch, overwrite = narFileOverwrite)
    

def writeMaterial(i, indFName, materialName, materialFilePath):
    materialStr = [
        'material ' + str(i), 
        '    name = ' + materialName, 
        '    optical', 
        '        nr = ' + str(1) if materialName == 'Air' else '        nr = userreal("' + materialFilePath + '",free_space_wavelength)', 
        '        ni = ' + str(0) if materialName == 'Air' else '        ni = userimag("' + materialFilePath + '",free_space_wavelength)', 
        '   end optical', 
        'end material', 
    ]
    with open(indFName, 'a') as foh:
        foh.write('\n'.join(materialStr))
        foh.write('\n\n')


def writeUserTaper(indFName, i = 1, expression = '1-(z-0.5)^2*4'):
    userTaperStr = [
        'user_taper ' + str(i),
        '    type = UF_EXPRESSION',
        '    expression = ' + expression,
        'end user_taper',
    ]
    with open(indFName, 'a') as foh:
        foh.write('\n'.join(userTaperStr))
        foh.write('\n\n')


def writeLens(indFName, i, rfrontStr,        # rfront
                            rbackStr,         # rback
                            tcenterStr,     # tcenter
                            ZlensStr, 
                            phiStr, 
                            refSegmentNum, 
                            WstarStr, 
                            draw_priority = 2, mask_layer = 3704, color = 7, begin_height = None):
    lensStr = [
        'lens ' + str(i), 
        '   draw_priority = ' + str(draw_priority), 
        '   mask_layer = ' + str(mask_layer), 
        '   color = ' + str2(color), 
        '   rfront = ' + rfrontStr,
        '   rback = ' + rbackStr, 
        '   tcenter = ' + tcenterStr, 
        '   angle = ' + phiStr, 
        '   begin.x = ' + ZlensStr + '*sin(' + phiStr + ') rel begin segment ' + str(refSegmentNum), 
        '   begin.z = ' + ZlensStr + '*cos(' + phiStr + ') rel begin segment ' + str(refSegmentNum), 
        '   begin.height = ' + begin_height if begin_height != None else '', 
        '   begin.width = ' + WstarStr, 
    ]
    lensStr = [x for x in lensStr if x]
    lensStr.append('end lens')
    with open(indFName, 'a') as foh:
        foh.write('\n'.join(lensStr))
        foh.write('\n\n')
    return i+1


def writeSegment(indFName, i, begin_x = None, begin_y = None, begin_z = None, end_x = None, end_y = None, end_z = None, 
                                orientation = 1, draw_priority = 2, cld = 0, 
                                width_taper = None, position_taper = 0, color = 12, mask_layer = 7100, 
                                begin_width = None, end_width = None, profile_type = 1, 
                                arc_type = None, arc_radius = None, arc_iangle = None, arc_fangle = None):
    position_taperStr = [0, 'TAPER_LINEAR', 'TAPER_ARC']
    segmentStr = [
        'segment ' + str(i), 
        '   profile_type = PROF_INACTIVE' if profile_type == 0 else '', 
        '   draw_priority = ' + str(draw_priority), 
        '   mask_layer = ' + str(mask_layer), 
        '   color = ' + str2(color), 
        '   orientation = ' + str(orientation), 
        '   position_taper = ' + position_taperStr[position_taper] if position_taper !=0 else '', 
        '   begin.x = ' + str2(begin_x) if begin_x != None else '', 
        '   begin.y = ' + str2(begin_y) if begin_y != None else '', 
        '   begin.z = ' + str2(begin_z) if begin_z != None else '', 
        '   begin.width = ' + str2(begin_width) if begin_width != None else '', 
        '   end.width = ' + str2(end_width) if end_width != None else '', 
        '   arc_type = ' + str2(arc_type) if arc_type != None else '', 
        '   arc_radius = ' + str2(arc_radius) if arc_radius != None else '', 
        '   arc_iangle = ' + str2(arc_iangle) if arc_iangle != None else '', 
        '   arc_fangle = ' + str2(arc_fangle) if arc_fangle != None else '', 
        '   width_taper = ' + str2(width_taper) if width_taper != None else '', 
        '   end.x = ' + str2(end_x) if end_x != None else '', 
        '   end.y = ' + str2(end_y) if end_y != None else '', 
        '   end.z = ' + str2(end_z) if end_z != None else '', 
    ]
    segmentStr = [x for x in segmentStr if x]
    segmentStr.append('end segment')
    with open(indFName, 'a') as foh:
        foh.write('\n'.join(segmentStr))
        foh.write('\n\n')
    if cld == 1:
        segmentStr = [
            'segment ' + str(i+1), 
            '   profile_type = PROF_INACTIVE' if profile_type == 0 else '', 
            # '   draw_priority = ' + str(draw_priority), 
            '   draw_priority = ' + str(draw_priority-2), 
            '   mask_layer = 3705', 
            # '   color = ' + str2(color), 
            '   color = 14', 
            '   orientation = ' + str(orientation), 
            '   position_taper = ' + position_taperStr[position_taper] if position_taper !=0 else '', 
            '   begin.x = ' + str2(begin_x) if begin_x != None else '', 
            '   begin.y = ' + str2(begin_y) if begin_y != None else '', 
            '   begin.z = ' + str2(begin_z) if begin_z != None else '', 
            '   begin.width = ' + str2(begin_width) + '+excess_width_cld' if begin_width != None else '   begin.width = width+excess_width_cld', 
            '   end.width = ' + str2(end_width) + '+excess_width_cld' if end_width != None else '   end.width = width+excess_width_cld', 
            '   arc_type = ' + str2(arc_type) if arc_type != None else '', 
            '   arc_radius = ' + str2(arc_radius) if arc_radius != None else '', 
            '   arc_iangle = ' + str2(arc_iangle) if arc_iangle != None else '', 
            '   arc_fangle = ' + str2(arc_fangle) if arc_fangle != None else '', 
            '   width_taper = ' + str2(width_taper) if width_taper != None else '', 
            '   end.x = ' + str2(end_x) if end_x != None else '', 
            '   end.y = ' + str2(end_y) if end_y != None else '', 
            '   end.z = ' + str2(end_z) if end_z != None else '', 
        ]
        segmentStr = [x for x in segmentStr if x]
        segmentStr.append('end segment')
        with open(indFName, 'a') as foh:
            foh.write('\n'.join(segmentStr))
            foh.write('\n\n')
        return i+2
    return i+1


def AWG_rsoftInd_generate(indFName, naFName, narFName, core_material, clad_material, Nin, Nout, Nchan, M, shape, 
                                                Di, Do, Lambda0, DLambda, Wit, Wot, Width, Width_fat, arc_radius, inputCenter, 
                                                DL_pitch, arcend_pitch, port_pitch, CLD = 1, excess_width_cld = 4, onlyParam = 0, Ltfat = 10, 
                                                Lit = '(Nslab_design/Lambda0)*(Wit*Wit)', Lot = '(Nslab_design/Lambda0)*(Wot*Wot)', 
                                                narFileOverwrite = 0, nslabFName = None, height = 0.22):
    
    writeBaiscParam(fileName = indFName, naFName = naFName, narFName = narFName, nslabFName = nslabFName, 
                                nfFName = core_material, nsubFName = clad_material, ncFName = clad_material, height = height, 
                                Nin = Nin, Nout = Nout, Nchan = Nchan, M = M, shape = shape, Lit = Lit, Lot = Lot, Ltfat = Ltfat, 
                                Di = Di, Do = Do, Lambda0 = Lambda0, DLambda = DLambda, narFileOverwrite = narFileOverwrite, 
                                Wit = Wit, Wot = Wot, Width = Width, Width_fat = Width_fat, arc_radius = arc_radius, inputCenterx = inputCenter[0], inputCenterz = inputCenter[1], 
                                DL_pitch = DL_pitch, arcend_pitch = arcend_pitch, port_pitch = port_pitch, excess_width_cld = excess_width_cld, 
                                )

    # material          # does matter
    materialName = ['Air', 'SiO2', 'Core']     # dont change
    materialFilePath = [None, clad_material if clad_material != None else '<rsoftmat>\SiO2_nk.dat', core_material if core_material != None else '<rsoftmat>\Si_nk.dat'] # doesnt matter if nslabFName is assigned
    for i in range(3):
        writeMaterial(i+1, indFName, materialName[i], materialFilePath[i])


    # userTaper
    writeUserTaper(indFName, 1)

    if onlyParam == 0:
        # right input lens
        writeLens(indFName, 1, 'Ri', '-Ro', 'Lstar', 'Zlens', '0', 2, 'Wstar', draw_priority = -3, color = 12)

        # color table:
        # 9 for light blue; 12 for light red; 10 for light green; 13 for light magenta; 11 for light cyan;


        # reference segment for right lens; begin: lower z; end: higher z.
        idxSeg = writeSegment(indFName, 2, begin_x = inputCenter[0], begin_z = inputCenter[1], 
                                            end_x = '0 rel begin segment 2', 
                                            end_z = 'Ro rel begin segment 2', 
                                            color = 10, profile_type = 0, 
                                            draw_priority = 10, mask_layer = 1002, orientation = 1)

        # right input lens, FC
        idxSeg = writeLens(indFName, idxSeg, 'Ririb', '-Rorib', 'Lstarrib', 'Zlensrib', '0', 2, 'Wstarrib', draw_priority = -4, mask_layer = 3505, color = 7)

        # right input lens, floor
        idxSeg = writeLens(indFName, idxSeg, 'Riflr', '-Roflr', 'Lstarflr', 'Zlensflr', '0', 2, 'Wstarflr', draw_priority = -5, mask_layer = 3705, color = 6)

        # input port taper
        idxSegList0 = []
        for i in range(Nin):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro-Lim)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                begin_z = '(Ro-Lim)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_x = '(Ro+Lit)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_z = '(Ro+Lit)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                begin_width = 'Wim', end_width = 'width', cld = CLD, 
                                                width_taper = 'width_taper_in', position_taper = 1, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)

            # input port taper, FC
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro-Lim)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                begin_z = '(Ro-Lim)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_x = '(Ro+Lit)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_z = '(Ro+Lit)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                begin_width = 'Wim*2', end_width = 'width', 
                                                width_taper = 'width_taper_in', position_taper = 1, 
                                                draw_priority = 1, mask_layer = 3505, orientation = 1, color = 7)

            # input port straight, for boolean
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro+Lit)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                begin_z = '(Ro+Lit)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_x = '(Ro+Lit+1)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_z = '(Ro+Lit+1)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                width_taper = 'TAPER_LINEAR', begin_width = 'width', end_width = 'width*2', 
                                                draw_priority = 3, mask_layer = 3507, orientation = 1, color = 0)

            # input port straight
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro+Litm)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                begin_z = '(Ro+Litm)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_x = '(Ro+Li)*sin(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', 
                                                end_z = '(Ro+Li)*cos(180-(180-zAi' + str2(i+1) + ')/2) rel end segment 2', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
            idxSegList0.append(idxSeg-1)
            
            # input arc segment
            if Nin%2 != 1 or i != Nin//2:
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                                begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                                arc_type = 'ARC_FREE', arc_radius = 'abs(Ri' + str2(i+1) + ')',
                                                arc_iangle = '180-(180-zAi' + str2(i+1) + ')/2', arc_fangle = '180', 
                                                position_taper = 2, cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)
                idxSegList0[i] = idxSeg-1
            
        # input port strengthen to constant z
        for i in range(Nin):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel end segment ' + str2(idxSegList0[round(Nin/2)]), 
                                            cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)     # light red

        # input arrayed taper
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_z = '(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_x = '(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_z = '(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_width = 'Wom', end_width = 'width', cld = CLD, 
                                            width_taper = 'width_taper_out', position_taper = 1, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
            
            # input arrayed taper, FC
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_z = '(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_x = '(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_z = '(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_width = 'Wom*2', end_width = 'width', 
                                            width_taper = 'width_taper_out', position_taper = 1, 
                                            draw_priority = 1, mask_layer = 3505, orientation = 1, color = 7)

        # input arrayed straight
        idxSegList0 = []
        for i in range(M):
            # straight
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro+Lotm)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_z = '(Ro+Lotm)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_x = '(Ro+Lo)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_z = '(Ro+Lo)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)      # light red
            
            idxSegList0.append(idxSeg-1)
            
            # straight, for boolean
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_z = '(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_x = '(Ro+Lot+1)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            end_z = '(Ro+Lot+1)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                            begin_width = 'width', end_width = 'width*2', width_taper = 'TAPER_LINEAR', 
                                            draw_priority = 3, mask_layer = 3507, orientation = 1, color = 0)

        # right1 arc segment
        # Rai = ((i-Asa)*arcend_pitch-(Ro+Lo)*sin(zAai))  /  (1-cos(zAai))
        idxSegList1 = []
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            arc_type = 'ARC_FREE', arc_radius = 'abs(Ra' + str2(i+1) + ')', 
                                            arc_iangle = 'zAa' + str2(i+1), arc_fangle = 0, 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)
            idxSegList1.append(idxSeg-1)

        # right1 straight to the constant z
        idxSegList0 = []
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), cld = CLD, 
                                            end_z = 'arcStraightBuffer' + str(i+1) + ' rel begin segment ' + str2(idxSeg), 
                                            # end_z = '(arcStraightBuffer'+str2(round(M/2))+'-arcStraightBuffer' + str(i+1) + ') rel begin segment ' + str2(idxSeg), 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
            idxSegList0.append(idxSeg-1)

        # right1 vertical waveguide
        idxSegList1 = []
        for i in range(M):
            # transition
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = 'Ltfat rel begin segment ' + str2(idxSeg), 
                                            begin_width = 'width', end_width = 'width_2', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            
            # straight fat
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                            end_z = '(DL' + str2(i+1) + '-DL1)/2+eq(shape,2)*' + str2(i) + '*DL_pitch rel begin segment ' + str(idxSeg), 
                            begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)   # blue
            
            # transition
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = 'Ltfat rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            idxSegList1.append(idxSeg-1)

        # right1 90-degree bending waveguide
        idxSegList0 = []
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '-bendingOffset rel end segment ' + str2(idxSegList1[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            arc_type = 'ARC_FREE', arc_radius = 'arc_radius', 
                                            arc_iangle = '0', arc_fangle = '-90', 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)  # light green
            idxSegList0.append(idxSeg-1)

        # right1 horizontal waveguide
        idxSegList1 = []
        for i in range(M):
            # transition
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = 'bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                            end_x = '-Ltfat rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel begin segment ' + str2(idxSeg), 
                                            begin_width = 'width', end_width = 'width_2', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan

            # straight fat
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '-(Lh_min+' + str(i) + '*(arcend_pitch+(eq(shape,1)*arcend_pitch+eq(shape,2)*DL_pitch))) rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel begin segment ' + str2(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9) # light blue
            
            # transition
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '-Ltfat rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel begin segment ' + str2(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            idxSegList1.append(idxSeg-1)

        # right2 90-degree bending waveguide
        idxSegList0 = []
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            begin_z = '-bendingOffset rel end segment ' + str2(idxSegList1[i]), 
                                            arc_type = 'ARC_FREE', arc_radius = 'arc_radius', 
                                            arc_iangle = '-90', arc_fangle = '-180', 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)  # light green
            idxSegList0.append(idxSeg-1)

        # right2 vertical waveguide
        idxSegList1 = []
        for i in range(M):
            # transition
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '-bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '-Ltfat rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width', end_width = 'width_2', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            
            # straight fat
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '-((DL' + str2(i+1) + '-DL1)/2+eq(shape,2)*(M-1)*DL_pitch) rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)  # light blue
            
            # transition
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '-Ltfat rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            idxSegList1.append(idxSeg-1)

        if shape == 2:
            # right3 90-degree bending waveguide
            idxSegList0 = []
            for i in range(M):
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '-bendingOffset rel end segment ' + str2(idxSegList1[i]), 
                                                begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                                arc_type = 'ARC_FREE', arc_radius = 'arc_radius', 
                                                arc_iangle = '-180', arc_fangle = '-90', 
                                                position_taper = 2, cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)  # light green
                idxSegList0.append(idxSeg-1)
            
            # right2 horizontal waveguide
            idxSegList1 = []
            for i in range(M):
                # transition
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                                begin_z = '-bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                                end_x = '-Ltfat rel begin segment ' + str2(idxSeg), 
                                                end_z = '0 rel begin segment ' + str(idxSeg), 
                                                begin_width = 'width', end_width = 'width_2', 
                                                width_taper = 'TAPER_LINEAR', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
                
                # straight fat
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                                begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                                end_x = '-(Lh_min+' + str(M-1-i) + '*(arcend_pitch+DL_pitch)) rel begin segment ' + str2(idxSeg), 
                                                end_z = '0 rel begin segment ' + str2(idxSeg), 
                                                begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)  # light blue
                
                # transition
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                                begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                                end_x = '-Ltfat rel begin segment ' + str2(idxSeg), 
                                                end_z = '0 rel begin segment ' + str(idxSeg), 
                                                begin_width = 'width_2', end_width = 'width', 
                                                width_taper = 'TAPER_LINEAR', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
                idxSegList1.append(idxSeg-1)
            
            # right4 90-degree bending waveguide
            idxSegList0 = []
            for i in range(M):
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                                begin_z = 'bendingOffset rel end segment ' + str2(idxSegList1[i]), 
                                                arc_type = 'ARC_FREE', arc_radius = 'arc_radius', 
                                                arc_iangle = '-90', arc_fangle = '-0', 
                                                position_taper = 2, cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)  # light green
                idxSegList0.append(idxSeg-1)

            # right3 vertical waveguide
            idxSegList1 = []
            for i in range(M):
                # transition
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '-bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                                begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                                end_x = '0 rel begin segment ' + str2(idxSeg), 
                                                end_z = 'Ltfat rel begin segment ' + str(idxSeg), 
                                                begin_width = 'width', end_width = 'width_2', 
                                                width_taper = 'TAPER_LINEAR', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
                
                # straight fat
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                                begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                                end_x = '0 rel begin segment ' + str2(idxSeg), 
                                                end_z = str2(M-1-i) + '*DL_pitch rel begin segment ' + str2(idxSeg), 
                                                begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)  # light blue
                
                # transition
                idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                                begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                                end_x = '0 rel begin segment ' + str2(idxSeg), 
                                                end_z = 'Ltfat rel begin segment ' + str(idxSeg), 
                                                begin_width = 'width_2', end_width = 'width', 
                                                width_taper = 'TAPER_LINEAR', cld = CLD, 
                                                draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
                idxSegList1.append(idxSeg-1)
            
        # constant z to the points before arc
        idxSegList0 = []
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '-outputSign*(arcStraightBuffer' + str(i+1) + ') rel begin segment ' + str2(idxSeg), 
                                            cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)     # light red
            idxSegList0.append(idxSeg-1)
                                
        # left arc segment
        # Rai = ((i-Asa)*arcend_pitch-(Ro+Lo)*sin(zAai))  /  (1-cos(zAai))
        idxSegList1 = []
        for i in range(M):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            arc_type = 'ARC_FREE', arc_radius = 'abs(Ra' + str2(i+1) + ')', 
                                            arc_iangle = '(1+outputSign)*90', arc_fangle = '(1+outputSign)*90-zAa' + str(i+1), 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)      # light green
            idxSegList1.append(idxSeg-1)

        # left lens
        idxRef_Ro = writeLens(indFName, idxSeg, 'Ri', '-Ro', 'Lstar', 'Zlens', '(1-outputSign)*90', idxSeg+1, 'Wstar', draw_priority = -3, color = 12)

        # left reference segments
        idxSeg = writeSegment(indFName, idxRef_Ro, 'outputSign*(Ro+Lo)*sin(zAa' + str2(M) + ') rel end segment ' + str(idxSegList1[-1]),     # ref to the last arc end
                                begin_z = '-outputSign*(Ro+Lo)*cos(zAa' + str2(M) + ') rel end segment ' + str(idxSegList1[-1]), 
                                end_x = '0 rel begin segment ' + str(idxRef_Ro), 
                                end_z = 'outputSign*Ro rel begin segment ' + str(idxRef_Ro), 
                                color = 10, profile_type = 0, 
                                draw_priority = 10, mask_layer = 1002, orientation = 1)

        # left lens, FC
        idxSeg = writeLens(indFName, idxSeg, 'Ririb', '-Rorib', 'Lstarrib', 'Zlensrib', '(1-outputSign)*90', idxSeg-1, 'Wstarrib', draw_priority = -4, mask_layer = 3505, color = 7)

        # left lens, floor
        idxSeg = writeLens(indFName, idxSeg, 'Riflr', '-Roflr', 'Lstarflr', 'Zlensflr', '(1-outputSign)*90', idxSeg-2, 'Wstarflr', draw_priority = -5, mask_layer = 3705, color = 6)

        # output arrayed region
        for i in range(M):
            # taper
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '-outputSign*(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            begin_z = 'outputSign*(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_x = '-outputSign*(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_z = 'outputSign*(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            width_taper = 'width_taper_out', position_taper = 1, cld = CLD, 
                                            begin_width = 'Wom', draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)         # light red
            
            # taper, FC
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '-outputSign*(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            begin_z = 'outputSign*(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_x = '-outputSign*(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_z = 'outputSign*(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            width_taper = 'width_taper_out', position_taper = 1, 
                                            begin_width = 'Wom*2', draw_priority = 1, mask_layer = 3505, orientation = 1, color = 7)       # yellow
            
            # straight
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '-outputSign*(Ro+Lotm)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            begin_z = 'outputSign*(Ro+Lotm)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_x = '-outputSign*(Ro+Lo)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_z = 'outputSign*(Ro+Lo)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)           # light red
            
            # straight, for boolean
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '-outputSign*(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            begin_z = 'outputSign*(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_x = '-outputSign*(Ro+Lot+1)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            end_z = 'outputSign*(Ro+Lot+1)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                            begin_width = 'width', end_width = 'width*2', width_taper = 'TAPER_LINEAR', 
                                            draw_priority = 3, mask_layer = 3507, orientation = 1, color = 0)           # light red

        # output ports
        idxSegList0 = []
        for i in range(Nout):
            # taper
            idxSeg = writeSegment(indFName, idxSeg, 
                                begin_x = '(Ro-Lim)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                begin_z = '(Ro-Lim)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                end_x = '(Ro+Lit)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                end_z = '(Ro+Lit)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                width_taper = 'width_taper_in', position_taper = 1, cld = CLD, 
                                begin_width = 'Wim', draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)         # light red
            
            # taper, FC
            idxSeg = writeSegment(indFName, idxSeg, 
                                begin_x = '(Ro-Lim)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                begin_z = '(Ro-Lim)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                end_x = '(Ro+Lit)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                end_z = '(Ro+Lit)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                width_taper = 'width_taper_in', position_taper = 1, 
                                begin_width = 'Wim*2', draw_priority = 1, mask_layer = 3505, orientation = 1, color = 7)       # yellow
            
            #straight
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro+Litm)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            begin_z = '(Ro+Litm)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            end_x = '(Ro+Li)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            end_z = '(Ro+Li)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
            idxSegList0.append(idxSeg-1)
            
            #straight, for boolean
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '(Ro+Lit)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            begin_z = '(Ro+Lit)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            end_x = '(Ro+Lit+1)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            end_z = '(Ro+Lit+1)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                            begin_width = 'width', end_width = 'width*2', width_taper = 'TAPER_LINEAR', 
                                            draw_priority = 3, mask_layer = 3507, orientation = 1, color = 0)

        # output arc for a pitch
        # Roi = (  (i-Aso)*port_pitch-(Ro+Li)*sin((180-zAoi)/2)  ) /  (1-cos((180-zAoi)/2))
        idxSegList1 = []
        for i in range(Nout):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            arc_type = 'ARC_FREE', arc_radius = 'abs(Ro' + str2(i+1) + ')', 
                                            arc_iangle = 'eq(shape,1)*180-outputSign*(180-zAo' + str2(i+1) + ')/2', arc_fangle = 'eq(shape,1)*180', 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
            idxSegList1.append(idxSeg-1)

        # output port strengthen to constant z
        for i in range(Nout):
            idxSeg = writeSegment(indFName, idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel end segment ' + str2(idxSegList1[round(Nout/2)]), 
                                            cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)     # light red