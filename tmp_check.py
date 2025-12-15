import sys, importlib, torch
print('python', sys.version)
print('torch', torch.__version__, 'xpu', hasattr(torch,'xpu') and torch.xpu.is_available(), 'cuda', torch.cuda.is_available())
mods=['intel_extension_for_pytorch','torch_directml','whisper','cv2','numpy','soundfile']
for mod in mods:
    try:
        m=importlib.import_module(mod)
        extra=''
        if mod=='intel_extension_for_pytorch':
            extra=f' xpu_available={getattr(m,"xpu",None) and m.xpu.is_available()}'
        if mod=='torch_directml':
            extra=f' devices={m.device_count()}'
        print(mod, 'ok', extra)
    except Exception as e:
        print(mod, 'FAILED', e)
