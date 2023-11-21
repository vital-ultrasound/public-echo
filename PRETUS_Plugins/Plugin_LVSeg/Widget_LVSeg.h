#pragma once
#include <QWidget>
#include <ifindImage.h>
#include <QtPluginWidgetBase.h>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QChart>
#include <boost/circular_buffer.hpp>

class QLabel;
class QCheckBox;
class QPushButton;


//class QSlider;

class Widget_LVSeg : public QtPluginWidgetBase
{
    Q_OBJECT

public:

    Widget_LVSeg(QWidget *parent = nullptr, Qt::WindowFlags f = Qt::WindowFlags());
    virtual void SendImageToWidgetImpl(ifind::Image::Pointer image);

    QCheckBox *mShowOverlayCheckbox;
    //QSlider *mSlider;
private:
    // raw pointer to new object which will be deleted by QT hierarchy
    QLabel *mLabel;
    QPushButton *mSaveButton;
    QtCharts::QLineSeries *mAreaSeries;
    QtCharts::QScatterSeries *mEDSeries;
    QtCharts::QScatterSeries *mESSeries;
    QtCharts::QChart *mChart;
    bool mIsBuilt;
    float mTime;

    double mTimeWindowWidth;
    boost::circular_buffer<double> mLastValues;

    /**
     * @brief Build the widget
     */
    void Build();

};
