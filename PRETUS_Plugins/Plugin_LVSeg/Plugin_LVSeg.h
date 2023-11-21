#pragma once

#include <Plugin.h>
#include "Worker_LVSeg.h"
#include "Widget_LVSeg.h"
#include <QtVTKVisualization.h>

class Plugin_LVSeg : public Plugin {
    Q_OBJECT

public:
    typedef Worker_LVSeg WorkerType;
    typedef Widget_LVSeg WidgetType;
    typedef QtVTKVisualization ImageWidgetType;
    Plugin_LVSeg(QObject* parent = 0);

    QString GetPluginName(void){ return "LV Seg";}
    QString GetPluginDescription(void) {return "Automatic Left Ventricle segmentation using Unet";}
    void SetCommandLineArguments(int argc, char* argv[]);

    void Initialize(void);

protected:
    virtual void SetDefaultArguments();

    template <class T> QString VectorToQString(std::vector<T> vec);
    template <class T> std::vector<T> QStringToVector(QString str);

public Q_SLOTS:
    virtual void slot_configurationReceived(ifind::Image::Pointer image);


};
