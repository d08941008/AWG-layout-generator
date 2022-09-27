import numpy as np
import os

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

def exportRaFile(naFName, nsFName, narFName, Lambda0, DLambda, Di, Do, Nchan, M, Md, Wot, arcend_pitch, overwrite = 0):
    na_data = read_txt(naFName, startLine = 0)
    na = np.vectorize(    lambda wl: np.interp(wl, na_data[:, 0], na_data[:, 1])    )  # red shift to calibrate the expriment data
    ng = np.vectorize(    lambda wl: na(wl) - wl*(na(wl+0.0001)-na(wl-0.0001))/(0.0002)    )  # red shift to calibrate the expriment data
    
    ns_data = read_txt(nsFName, startLine = 0)
    ns = np.vectorize(    lambda wl: np.interp(wl, ns_data[:, 0], ns_data[:, 1])    )  # red shift to calibrate the expriment data
    
    FSR = Nchan * DLambda
    mp = Lambda0 / FSR
    DispersionFactor = ng(Lambda0) / na(Lambda0)
    m = round(mp / DispersionFactor)
    Ro = ns(Lambda0)*Di*Do / (m*DispersionFactor*DLambda)
    Ao = Do/Ro
    Asa = (M+2*Md+1)/2
    Lot = ns(Lambda0)/Lambda0*Wot**2
    Lo = 2*Lot
    idxRa = np.linspace(1, M, M)
    zAai = (idxRa-Asa)*Ao
    Rai = ((idxRa-Asa)*arcend_pitch - (Ro+Lo)*np.sin(zAai)) / (1-np.cos(zAai))
    numRai = round(np.size(Rai, 0)/2)
    Rai_data = np.transpose(np.vstack([abs(Rai[0:numRai]), np.ones(numRai)]))
    print(Rai)
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
                                    Lot = '(Nslab_design/Lambda0)*(Wot^2)', 
                                    arcend_pitch = 2.6, 
                                    DL_pitch = 4, 
                                    arcend_straightBuffer = 0, 
                                    arc_radius = 10, 
                                    arc_bendingOffset = 0, 
                                    # Lvertical_min = 10, 
                                    Lh_min = 0.1, 
                                    Width_fat = None,       #
                                    Nd = 0,                     # reference number of the array region for differential calculation
                                    Md = 0, 
                                    naFName = 'na.txt',        # effective index of the array waveguide with the excess length DL (delta_L)
                                    narFName = 'naR.txt', narFileOverwrite = 0,  
                                    nsFName = 'ns.txt', 
                                    inputCenterx = 1000, 
                                    inputCenterz = 100, 
                                    nDL = 2,        # minimum amount of DL for DLi, can be considered as the minimum straight length for all AWs
                                    port_pitch = 3, 
                                    excess_width_cld = 4,       # trench width, in um
                                    shape = 2,                          # 0 for smit; 1 for rectangular; 2 for S-shaped;
                                    ):
    basicParamStr = [
        'Ai = (Di/Ri)*(180/pi)',                        # pitch angle of input/output waveguides, inner circle, in degree
        'Ao = (Do/Ro)*(180/pi)',                    # pitch angle of input/output waveguides, outer circle, in degree
        'Asa = (M+2*Md+1)/2',                       # middle index of the arrayed waveguides (can be double)
        'Asi = (Nin+2*Nd+1)/2',                     # middle index of the input waveguides (can be double)
        'Aso = (Nout+2*Nd+1)/2',                    # middle index of the output wavegudies (can be double)
        'CAi = sinc(Ai/2)',                                 #  
        'CAo = sinc(Ao/2)',                             # 
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
        'Dz = 0',
        'GratingOrder = round(Lambda0/(Nchan*DLambda*DispersionFactor))',
        'HasNaGF = defined("Nguide_file")',
        'HasNsGF = defined("Neffslab_file")',
        'Hside = Design_slab_height',
        'Lambda0 = ' + str(Lambda0),
        'Li = 2*Lit',
        'Lic = Ri*(1-cos(Ai/2))',
        'Lim = max(Ro/100,Lic)',
        'Lit = (Nslab_design/Lambda0)*(Wit^2)',
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
        'Neffslab_file = ' + nsFName, 
        'Ri = Ro/2',
        'Ro = (Nslab_design/GratingOrder)*((Di*Do)/(DLambda*DispersionFactor))',
        'Sagi = Ri-sqrt(Ri^2-(Wstar/2)^2)',
        'Sago = Ro-sqrt(Ro^2-(Wstar/2)^2)',
        'W = width',
        'Wim = W+(Wit-W)*(Lit+Lim)/(Lit+Lic)',
        'Wit = ' + str(Wit),
        'Wom = W+(Wot-W)*(Lot+Lom)/(Lot+Loc)',
        'Wot = ' + str(Wot),
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
        'dimension = 3',
        'domain_delta = width',
        'domain_max = 300',
        'domain_min = 100',
        'domain_round_sym = 1',
        'eim = 0',
        'free_space_wavelength = Lambda0',
        # 'grid_size = 0.025',
        # 'grid_size_y = 0.02',
        'height = 0.22',
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
        'slab_material = Si',                                                       # doesnt matter
        'slice_display_mode = DISPLAY_CONTOURMAPXZ',
        'slice_output_format = OUTPUT_NONE',
        # 'step_size = 0.125',
        'structure = STRUCT_RIBRIDGE',
        'vector = 2',
        'width = ' + str(Width),        # width of the strip-loaded waveguide
        'width_2 = ' + str(1.5*Width) if Width_fat == None else 'width_2 = ' + str(Width_fat),
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
        'arc_bendingOffset = ' + str2(arc_bendingOffset), 
        'port_pitch = ' + str2(port_pitch), 
        'excess_width_cld = ' + str2(excess_width_cld), 
        'shape = ' + str2(shape), 
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
        basicParamStr.append('Ro' + str(i+1) + ' = ((' + str(i+1) + '-Aso)*port_pitch-(Ro+Lo)*sin((180-zAo' + str(i+1) + ')/2))/(1-cos((180-zAo' + str(i+1) + ')/2))')
    
    ## arc radius for input port
    for i in range(Nin):
        # basicParamStr.append('Ri' + str(i+1) + ' = ((' + str(i+1) + '-Asi)*port_pitch-(Ro+Li)*sin((180-zAi' + str(i+1) + ')/2))/(1-cos((180-zAi' + str(i+1) + ')/2))')
        if Nin%2 != 1 or i != Nin//2:
            basicParamStr.append('Ri' + str(i+1) + ' = ((' + str(i+1) + '-Asi)*port_pitch-(Ri+Li)*sin(180-zAi' + str(i+1) + '))/(1-cos(180-zAi' + str(i+1) + '))')
    
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
    
    exportRaFile(naFName, nsFName, narFName, Lambda0, DLambda, Di, Do, Nchan, M, Md, Wot, arcend_pitch, overwrite = narFileOverwrite)
    

def writeMaterial(i, materialName, materialFilePath):
    materialStr = [
        'material ' + str(i), 
        '    name = ' + materialName, 
        '    optical', 
        '        nr = ' + str(1) if materialName == 'Air' else '        nr = userreal("' + materialFilePath + '",free_space_wavelength)', 
        '        ni = ' + str(0) if materialName == 'Air' else '        ni = userimag("' + materialFilePath + '",free_space_wavelength)', 
        '   end optical', 
        'end material', 
    ]
    with open(fileName, 'a') as foh:
        foh.write('\n'.join(materialStr))
        foh.write('\n\n')


def writeUserTaper(i = 1, expression = '1-(z-0.5)^2*4'):
    userTaperStr = [
        'user_taper ' + str(i),
        '    type = UF_EXPRESSION',
        '    expression = ' + expression,
        'end user_taper',
    ]
    with open(fileName, 'a') as foh:
        foh.write('\n'.join(userTaperStr))
        foh.write('\n\n')


def writeLens(i, rfrontStr,        # rfront
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
    with open(fileName, 'a') as foh:
        foh.write('\n'.join(lensStr))
        foh.write('\n\n')
    return i+1


def writeSegment(i, begin_x = None, begin_y = None, begin_z = None, end_x = None, end_y = None, end_z = None, 
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
    with open(fileName, 'a') as foh:
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
        with open(fileName, 'a') as foh:
            foh.write('\n'.join(segmentStr))
            foh.write('\n\n')
        return i+2
    return i+1



##................................................ main ........................................................##
# Si
fileName = 'D:\\SheldonChung\\PhD\\AWG\\RSoft_AWG\\pythonWriteInd\\Si\\AWG_Si_test20220329.ind'
onlyParam, narFileOverwrite = 0, 0
Lambda0, DLambda = 1.3, 0.02
Width, Width_fat = 0.38, 0.68
Nin, Nout, Nchan, M = 3, 6, 6, 16
Di, Do, Wit, Wot = 1.9, 1.9, 1.7, 1.7
arc_radius = 10
arcend_pitch, DL_pitch, port_pitch = 2.6, 4, 4
shape = 2                          # 0 for smit; 1 for rectangular; 2 for S-shaped;
naFName = 'Neff_Si_test20220328.dat'        # test for silicon-based
nsFName = 'Si_nk.dat'
narFName = 'NeffR_Si_test20220328.dat'      # test fot silicon-based

# SiN
# fileName = 'D:\\SheldonChung\\PhD\\AWG\\RSoft_AWG\\pythonWriteInd\\SiN\\AWG_SiN_test20220329.ind'
# onlyParam = 0
# Lambda0, DLambda = 1.3, 0.02
# Width, Width_fat = 0.7, 0.9
# Nin, Nout, Nchan, M = 1, 4, 6, 16
# Di, Do, Wit, Wot = 5*Width, 5*Width, 5*Width-0.15, 5*Width-0.15
# arc_radius = 50
# arcend_pitch, DL_pitch, port_pitch = 5, 4, 4
# shape = 2                          # 0 for smit; 1 for rectangular; 2 for S-shaped;
# naFName = 'neff_SiNwidth900nm.dat'
# nsFName = 'SiN_material.dat'
# narFName = 'neffR_SiNwidth700nm.dat'
# narFileOverwrite = 0

# InP
# fileName = 'D:\\SheldonChung\\PhD\\AWG\\RSoft_AWG\\pythonWriteInd\\AWG_InP_test20220328OK.ind'
# Lambda0, DLambda = 1.55092, 0.0032
# Width, Width_fat = 1.8, 1.8
# Nin, Nout, Nchan, M = 1, 4, 6, 16
# Di, Do, Wit, Wot = 3.3, 3.3, 1.8, 1.8
# arc_radius = 50
# arcend_pitch, DL_pitch, port_pitch = 4, 4, 4
# shape = 1                          # 0 for smit; 1 for rectangular; 2 for S-shaped;
# naFName = 'neff_deep_sweep.txt'        # test for silicon-based
# nsFName = 'neffslab_shallow.txt'
# narFName = 'neffR_deep.txt'      # test fot silicon-based


inputCenter = (1000, 100)   # interface center between input port and star coupler, to avoid the GDS-II export bug

CLD = 1     # for drawing CLD
excess_width_cld = 4       # trench width, in um
##..........................................................................................................##
writeBaiscParam(fileName = fileName, naFName = naFName, narFName = narFName, nsFName = nsFName, 
                            Nin = Nin, Nout = Nout, Nchan = Nchan, M = M, shape = shape, 
                            Di = Di, Do = Do, Lambda0 = Lambda0, DLambda = DLambda, narFileOverwrite = narFileOverwrite, 
                            Wit = Wit, Wot = Wot, Width = Width, Width_fat = Width_fat, arc_radius = arc_radius, 
                            DL_pitch = DL_pitch, arcend_pitch = arcend_pitch, port_pitch = port_pitch, 
                            )


# material          # doesnt matter
materialName = ['Air', 'SiO2', 'Si']
materialFilePath = [None, '<rsoftmat>\SiO2_nk.dat', '<rsoftmat>\Si_nk.dat']
# materialFilePath = [None, '<rsoftmat>\SiO2_nk.dat', 'SiN_material.dat']
for i in range(3):
    writeMaterial(i+1, materialName[i], materialFilePath[i])

# userTaper
writeUserTaper(1)

if onlyParam == 0:
    # right input lens
    writeLens(1, 'Ri', '-Ro', 'Lstar', 'Zlens', '0', 2, 'Wstar', draw_priority = -3, color = 12)

    # color table:
    # 9 for light blue; 12 for light red; 10 for light green; 13 for light magenta; 11 for light cyan;


    # reference segment for right lens; begin: lower z; end: higher z.
    idxSeg = writeSegment(2, begin_x = inputCenter[0], begin_z = inputCenter[1], 
                                        end_x = 'Lsep*sin(0) rel begin segment 2', 
                                        end_z = 'Lsep*cos(0) rel begin segment 2', 
                                        color = 10, profile_type = 0, 
                                        draw_priority = 10, mask_layer = 1002, orientation = 1)

    # right input lens, FC
    idxSeg = writeLens(idxSeg, 'Ririb', '-Rorib', 'Lstarrib', 'Zlensrib', '0', 2, 'Wstarrib', draw_priority = -4, color = 7)

    # right input lens, floor
    idxSeg = writeLens(idxSeg, 'Riflr', '-Roflr', 'Lstarflr', 'Zlensflr', '0', 2, 'Wstarflr', draw_priority = -5, color = 6)

    # input port taper
    idxSegList0 = []
    for i in range(Nin):
        idxSeg = writeSegment(idxSeg, begin_x = '(Ri-Lim)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            begin_z = '(Ri-Lim)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_x = '(Ri+Lit)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_z = '(Ri+Lit)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            begin_width = 'Wim', end_width = 'width', cld = CLD, 
                                            width_taper = 'width_taper_in', position_taper = 1, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)

        # input port taper, FC
        idxSeg = writeSegment(idxSeg, begin_x = '(Ri-Lim)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            begin_z = '(Ri-Lim)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_x = '(Ri+Lit)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_z = '(Ri+Lit)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            begin_width = 'Wim*2', end_width = 'width', 
                                            width_taper = 'width_taper_in', position_taper = 1, 
                                            draw_priority = 1, mask_layer = 3505, orientation = 1, color = 7)

        # input port straight, for boolean
        idxSeg = writeSegment(idxSeg, begin_x = '(Ri+Lit)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            begin_z = '(Ri+Lit)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_x = '(Ri+Lit+1)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_z = '(Ri+Lit+1)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            width_taper = 'TAPER_LINEAR', begin_width = 'width', end_width = 'width*2', 
                                            draw_priority = 3, mask_layer = 3507, orientation = 1, color = 0)

        # input port straight
        idxSeg = writeSegment(idxSeg, begin_x = '(Ri+Litm)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            begin_z = '(Ri+Litm)*cos(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_x = '(Ri+Li)*sin(zAi' + str2(i+1) + ') rel end segment 2', 
                                            end_z = '(Ri+Li)*cos(zAi' + str2(i+1) + ') rel end segment 2', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
        idxSegList0.append(idxSeg-1)
        
        # input arc segment
        if Nin%2 != 1 or i != Nin//2:
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            arc_type = 'ARC_FREE', arc_radius = 'abs(Ri' + str2(i+1) + ')',
                                            # arc_iangle = '180-(180-zAi' + str2(i+1) + ')/2', arc_fangle = '180', 
                                            arc_iangle = '180-(180-zAi' + str2(i+1) + ')', arc_fangle = '180', 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)
            idxSegList0[i] = idxSeg-1
        
    # input port strengthen to constant z
    for i in range(Nin):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '0 rel end segment ' + str2(idxSegList0[round(Nin/2)]), 
                                        cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)     # light red

    # input arrayed taper
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_z = '(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_x = '(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_z = '(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_width = 'Wom', end_width = 'width', cld = CLD, 
                                        width_taper = 'width_taper_out', position_taper = 1, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
        
        # input arrayed taper, FC
        idxSeg = writeSegment(idxSeg, begin_x = '(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_z = '(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_x = '(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_z = '(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_width = 'Wom*2', end_width = 'width', 
                                        width_taper = 'width_taper_out', position_taper = 1, 
                                        draw_priority = 1, mask_layer = 3704, orientation = 1, color = 7)

    # input arrayed straight
    idxSegList0 = []
    for i in range(M):
        # straight
        idxSeg = writeSegment(idxSeg, begin_x = '(Ro+Lotm)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_z = '(Ro+Lotm)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_x = '(Ro+Lo)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_z = '(Ro+Lo)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)      # light red
        
        idxSegList0.append(idxSeg-1)
        
        # straight, for boolean
        idxSeg = writeSegment(idxSeg, begin_x = '(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_z = '(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_x = '(Ro+Lot+1)*sin(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        end_z = '(Ro+Lot+1)*cos(zAa' + str2(i+1) + ') rel begin segment 2', 
                                        begin_width = 'width', end_width = 'width*2', width_taper = 'TAPER_LINEAR', 
                                        draw_priority = 3, mask_layer = 3704, orientation = 1, color = 0)

    # right1 arc segment
    # Rai = ((i-Asa)*arcend_pitch-(Ro+Lo)*sin(zAai))  /  (1-cos(zAai))
    idxSegList1 = []
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        arc_type = 'ARC_FREE', arc_radius = 'abs(Ra' + str2(i+1) + ')', 
                                        arc_iangle = 'zAa' + str2(i+1), arc_fangle = 0, 
                                        position_taper = 2, cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)
        idxSegList1.append(idxSeg-1)

    # right1 straight to the constant z
    idxSegList0 = []
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
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
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '10 rel begin segment ' + str2(idxSeg), 
                                        begin_width = 'width', end_width = 'width_2', 
                                        width_taper = 'TAPER_LINEAR', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
        
        # straight fat
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                        begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                        end_z = '(DL' + str2(i+1) + '-DL1)/2+eq(shape,2)*' + str2(i) + '*DL_pitch rel begin segment ' + str(idxSeg), 
                        begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)   # blue
        
        # transition
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                        begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '10 rel begin segment ' + str(idxSeg), 
                                        begin_width = 'width_2', end_width = 'width', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
        idxSegList1.append(idxSeg-1)

    # right1 90-degree bending waveguide
    idxSegList0 = []
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '-arc_bendingOffset rel end segment ' + str2(idxSegList1[i]), 
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
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = 'arc_bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                        end_x = '-10 rel begin segment ' + str2(idxSeg), 
                                        end_z = '0 rel begin segment ' + str2(idxSeg), 
                                        begin_width = 'width', end_width = 'width_2', 
                                        width_taper = 'TAPER_LINEAR', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan

        # straight fat
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                        begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                        end_x = '-(Lh_min+' + str(i) + '*(arcend_pitch+(eq(shape,1)*arcend_pitch+eq(shape,2)*DL_pitch))) rel begin segment ' + str2(idxSeg), 
                                        end_z = '0 rel begin segment ' + str2(idxSeg), 
                                        begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9) # light blue
        
        # transition
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                        begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                        end_x = '-10 rel begin segment ' + str2(idxSeg), 
                                        end_z = '0 rel begin segment ' + str2(idxSeg), 
                                        begin_width = 'width_2', end_width = 'width', 
                                        width_taper = 'TAPER_LINEAR', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
        idxSegList1.append(idxSeg-1)

    # right2 90-degree bending waveguide
    idxSegList0 = []
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                        begin_z = '-arc_bendingOffset rel end segment ' + str2(idxSegList1[i]), 
                                        arc_type = 'ARC_FREE', arc_radius = 'arc_radius', 
                                        arc_iangle = '-90', arc_fangle = '-180', 
                                        position_taper = 2, cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)  # light green
        idxSegList0.append(idxSeg-1)

    # right2 vertical waveguide
    idxSegList1 = []
    for i in range(M):
        # transition
        idxSeg = writeSegment(idxSeg, begin_x = '-arc_bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '-10 rel begin segment ' + str(idxSeg), 
                                        begin_width = 'width', end_width = 'width_2', 
                                        width_taper = 'TAPER_LINEAR', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
        
        # straight fat
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                        begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '-((DL' + str2(i+1) + '-DL1)/2+eq(shape,2)*(M-1)*DL_pitch) rel begin segment ' + str(idxSeg), 
                                        begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)  # light blue
        
        # transition
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                        begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '-10 rel begin segment ' + str(idxSeg), 
                                        begin_width = 'width_2', end_width = 'width', 
                                        width_taper = 'TAPER_LINEAR', cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
        idxSegList1.append(idxSeg-1)

    if shape == 2:
        # right3 90-degree bending waveguide
        idxSegList0 = []
        for i in range(M):
            idxSeg = writeSegment(idxSeg, begin_x = '-arc_bendingOffset rel end segment ' + str2(idxSegList1[i]), 
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
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '-arc_bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                            end_x = '-10 rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width', end_width = 'width_2', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            
            # straight fat
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '-(Lh_min+' + str(M-1-i) + '*(arcend_pitch+DL_pitch)) rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel begin segment ' + str2(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)  # light blue
            
            # transition
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '-10 rel begin segment ' + str2(idxSeg), 
                                            end_z = '0 rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            idxSegList1.append(idxSeg-1)
        
        # right4 90-degree bending waveguide
        idxSegList0 = []
        for i in range(M):
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                            begin_z = 'arc_bendingOffset rel end segment ' + str2(idxSegList1[i]), 
                                            arc_type = 'ARC_FREE', arc_radius = 'arc_radius', 
                                            arc_iangle = '-90', arc_fangle = '-0', 
                                            position_taper = 2, cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)  # light green
            idxSegList0.append(idxSeg-1)

        # right3 vertical waveguide
        idxSegList1 = []
        for i in range(M):
            # transition
            idxSeg = writeSegment(idxSeg, begin_x = '-arc_bendingOffset rel end segment ' + str2(idxSegList0[i]), 
                                            begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '10 rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width', end_width = 'width_2', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            
            # straight fat
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = str2(M-1-i) + '*DL_pitch rel begin segment ' + str2(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width_2', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 9)  # light blue
            
            # transition
            idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSeg-1), 
                                            begin_z = '0 rel end segment ' + str2(idxSeg-1), 
                                            end_x = '0 rel begin segment ' + str2(idxSeg), 
                                            end_z = '10 rel begin segment ' + str(idxSeg), 
                                            begin_width = 'width_2', end_width = 'width', 
                                            width_taper = 'TAPER_LINEAR', cld = CLD, 
                                            draw_priority = 2, mask_layer = 3704, orientation = 1, color = 11)  # light cyan
            idxSegList1.append(idxSeg-1)
        
    # constant z to the points before arc
    idxSegList0 = []
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '-outputSign*(arcStraightBuffer' + str(i+1) + ') rel begin segment ' + str2(idxSeg), 
                                        cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)     # light red
        idxSegList0.append(idxSeg-1)
                            
    # left arc segment
    # Rai = ((i-Asa)*arcend_pitch-(Ro+Lo)*sin(zAai))  /  (1-cos(zAai))
    idxSegList1 = []
    for i in range(M):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        arc_type = 'ARC_FREE', arc_radius = 'abs(Ra' + str2(i+1) + ')', 
                                        arc_iangle = '(1+outputSign)*90', arc_fangle = '(1+outputSign)*90-zAa' + str(i+1), 
                                        position_taper = 2, cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 10)      # light green
        idxSegList1.append(idxSeg-1)

    # left lens
    idxRef_Ro = writeLens(idxSeg, 'Ri', '-Ro', 'Lstar', 'Zlens', '(1-outputSign)*90', idxSeg+1, 'Wstar', draw_priority = -3, color = 12)

    # left reference segments
    idxSeg = writeSegment(idxRef_Ro, 'outputSign*(Ro+Lo)*sin(zAa' + str2(M) + ') rel end segment ' + str(idxSegList1[-1]),     # ref to the last arc end
                            begin_z = '-outputSign*(Ro+Lo)*cos(zAa' + str2(M) + ') rel end segment ' + str(idxSegList1[-1]), 
                            end_x = '0 rel begin segment ' + str(idxRef_Ro), 
                            end_z = 'outputSign*Ro rel begin segment ' + str(idxRef_Ro), 
                            color = 10, profile_type = 0, 
                            draw_priority = 10, mask_layer = 1002, orientation = 1)

    # left lens, FC
    idxSeg = writeLens(idxSeg, 'Ririb', '-Rorib', 'Lstarrib', 'Zlensrib', '(1-outputSign)*90', idxSeg-1, 'Wstarrib', draw_priority = -4, color = 7)

    # left lens, floor
    idxSeg = writeLens(idxSeg, 'Riflr', '-Roflr', 'Lstarflr', 'Zlensflr', '(1-outputSign)*90', idxSeg-2, 'Wstarflr', draw_priority = -5, color = 6)

    # output arrayed region
    for i in range(M):
        # taper
        idxSeg = writeSegment(idxSeg, begin_x = '-outputSign*(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        begin_z = 'outputSign*(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_x = '-outputSign*(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_z = 'outputSign*(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        width_taper = 'width_taper_out', position_taper = 1, cld = CLD, 
                                        begin_width = 'Wom', draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)         # light red
        
        # taper, FC
        idxSeg = writeSegment(idxSeg, begin_x = '-outputSign*(Ro-Lom)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        begin_z = 'outputSign*(Ro-Lom)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_x = '-outputSign*(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_z = 'outputSign*(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        width_taper = 'width_taper_out', position_taper = 1, 
                                        begin_width = 'Wom*2', draw_priority = 1, mask_layer = 3704, orientation = 1, color = 7)       # yellow
        
        # straight
        idxSeg = writeSegment(idxSeg, begin_x = '-outputSign*(Ro+Lotm)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        begin_z = 'outputSign*(Ro+Lotm)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_x = '-outputSign*(Ro+Lo)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_z = 'outputSign*(Ro+Lo)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)           # light red
        
        # straight, for boolean
        idxSeg = writeSegment(idxSeg, begin_x = '-outputSign*(Ro+Lot)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        begin_z = 'outputSign*(Ro+Lot)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_x = '-outputSign*(Ro+Lot+1)*sin(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        end_z = 'outputSign*(Ro+Lot+1)*cos(zAa' + str2(i+1) + ') rel begin segment ' + str(idxRef_Ro), 
                                        begin_width = 'width', end_width = 'width*2', width_taper = 'TAPER_LINEAR', 
                                        draw_priority = 3, mask_layer = 3704, orientation = 1, color = 0)           # light red

    # output ports
    idxSegList0 = []
    for i in range(Nout):
        # taper
        idxSeg = writeSegment(idxSeg, 
                            begin_x = '(Ro-Lim)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            begin_z = '(Ro-Lim)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            end_x = '(Ro+Lit)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            end_z = '(Ro+Lit)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            width_taper = 'width_taper_in', position_taper = 1, cld = CLD, 
                            begin_width = 'Wim', draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)         # light red
        
        # taper, FC
        idxSeg = writeSegment(idxSeg, 
                            begin_x = '(Ro-Lim)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            begin_z = '(Ro-Lim)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            end_x = '(Ro+Lit)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            end_z = '(Ro+Lit)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                            width_taper = 'width_taper_in', position_taper = 1, 
                            begin_width = 'Wim*2', draw_priority = 1, mask_layer = 3704, orientation = 1, color = 7)       # yellow
        
        #straight
        idxSeg = writeSegment(idxSeg, begin_x = '(Ro+Litm)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        begin_z = '(Ro+Litm)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        end_x = '(Ro+Li)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        end_z = '(Ro+Li)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
        idxSegList0.append(idxSeg-1)
        
        #straight, for boolean
        idxSeg = writeSegment(idxSeg, begin_x = '(Ro+Lit)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        begin_z = '(Ro+Lit)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        end_x = '(Ro+Lit+1)*sin((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        end_z = '(Ro+Lit+1)*-outputSign*cos((180-zAo' + str(i+1) + ')/2) rel end segment ' + str(idxRef_Ro), 
                                        begin_width = 'width', end_width = 'width*2', width_taper = 'TAPER_LINEAR', 
                                        draw_priority = 3, mask_layer = 3704, orientation = 1, color = 0)

    # output arc for a pitch
    # Roi = (  (i-Aso)*port_pitch-(Ro+Lit+Lot)*sin((180-zAoi)/2)  ) /  (1-cos((180-zAoi)/2))
    idxSegList1 = []
    for i in range(Nout):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList0[i]), 
                                        arc_type = 'ARC_FREE', arc_radius = 'abs(Ro' + str2(i+1) + ')', 
                                        arc_iangle = 'eq(shape,1)*180-outputSign*(180-zAo' + str2(i+1) + ')/2', arc_fangle = 'eq(shape,1)*180', 
                                        position_taper = 2, cld = CLD, 
                                        draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)
        idxSegList1.append(idxSeg-1)

    # output port strengthen to constant z
    for i in range(Nout):
        idxSeg = writeSegment(idxSeg, begin_x = '0 rel end segment ' + str2(idxSegList1[i]), 
                                        begin_z = '0 rel end segment ' + str2(idxSegList1[i]), 
                                        end_x = '0 rel begin segment ' + str2(idxSeg), 
                                        end_z = '0 rel end segment ' + str2(idxSegList1[round(Nout/2)]), 
                                        cld = CLD, draw_priority = 2, mask_layer = 3704, orientation = 1, color = 12)     # light red