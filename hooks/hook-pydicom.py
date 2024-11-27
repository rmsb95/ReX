from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = collect_data_files('pydicom') + copy_metadata('pylibjpeg') + copy_metadata('pylibjpeg-libjpeg') + copy_metadata('pylibjpeg-openjpeg')
hiddenimports = ['pydicom.encoders.gdcm', 'pydicom.encoders.pylibjpeg', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg']
