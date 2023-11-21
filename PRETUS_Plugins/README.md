# Plug-in based, Real-time Ultrasound (PRETUS) Plugins for 4 chamber view
[PRETUS](https://github.com/gomezalberto/pretus) [Plugin_LVSeg](Plugin_LVSeg) contains the four chamber LV segmentation plug-in in real-time.

## Building plug-in
* Install PRETUS covering all dependencies and same packages versions:    
See https://github.com/gomezalberto/pretus 


* Open terminal with conda environment
``` 
conda activate pretus
```

* Creating building paths
``` 
mkdir -p $HOME/build/pretus/LVSeg/ && cd $HOME/build/pretus/LVSeg/ 
```

* Config PLUGIN_* PATHS
``` 
* /home/ag09/build/pretus/release
PLUGIN FOLDER: /home/ag09/build/pretus/release/lib

* /home/ag09/build/pretus/LVSeg
PLUGIN_INCLUDE_DIR ~/local/pretus/include
MAKE_INSTALL_PREFIX ~/local/pretus-vital-echo

```

* build pretus and plugins
``` 
* ~/build/pretus/release$ make
* ~/build/pretus/LVSeg$ make; make install
```

* Edit `~/.config/iFIND/PRETUS.conf`
```
[MainApp]
plugin_folder="/home/ag09/build/pretus/release/lib;/home/ag09/local/pretus-vital-echo/lib"
```

* Open cmake-gui
```
cd $HOME/repositories/echocardiography/source/PRETUS_Plugins
cmake-gui .
```

* Creating building paths
``` 
Source code: $HOME/repositories/echocardiography/source/PRETUS_Plugins
Where to build binaries: $HOME/build/pretus/LVSeg
```

* CMake tags in PRETUS
``` 
    CMAKE_INSTALL_PREFIX set to $HOME/local/pretus   (Press configure)
    PLUGIN_INCLUDE_DIR set to $HOME/local/pretus/include (Press configure)
    PLUGIN_LIBRARY set to $HOME/local/pretus/lib/libPlugin.so (Press configure)
    PLUGIN_LIBRARY_DIR set to $HOME/local/pretus/lib (Press configure)

    VTK_DIR set to $HOME/workspace/VTK/release  (Press configure)
    ITK_DIR set to $HOME/workspace/ITK/release (Press configure) 
    
    PYTHON_LIBRARY set to $HOME/anaconda3/envs/pretus/lib/libpython3.7m.so (Press configure) 
    PYTHON_INCLUDE_DIR set to $HOME/anaconda3/envs/pretus/include/python3.7m (Press configure) 
    pybind11_DIR set to $HOME/local/pybind11/share/cmake/pybind11 (Press configure) 
    
    Qt settings
        Qt5Concurrent_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Concurrent
        Qt5Core_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Core
        Qt5Gui_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Gui
        Qt5OpenGL_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5OpenGL
        Qt5PrintSupport_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5PrintSupport
        Qt5Sql_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Sql
        Qt5Widgets_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Widgets
        Qt5X11Extras_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5X11Extras
        Qt5Xml_DIR set to $HOME/Qt/5.12.5/gcc_64/lib/cmake/Qt5Xml
```

* Make project 
``` 
cd $HOME/build/pretus/LVSeg
conda activate pretus
make
make install 
```

