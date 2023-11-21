#include "Widget_LVSeg.h"
#include <QSlider>
#include <QLabel>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QPushButton>
#include <QtCharts/QChartView>
#include <QtCharts/QValueAxis>
#include <QRandomGenerator>
#include <chrono>
#include <boost/algorithm/minmax_element.hpp>
//#include "qcustomplot.h"

//#include "QtInfoPanelLabelConFetalWeight.h"

Widget_LVSeg::Widget_LVSeg(
        QWidget *parent, Qt::WindowFlags f)
    : QtPluginWidgetBase(parent, f)
{

    this->mWidgetLocation = WidgetLocation::top_right;
    mStreamTypes = ifind::InitialiseStreamTypeSetFromString("LVSeg");
    mIsBuilt = false;
    mTime = 0;
    mLabel = new QLabel("Text not set", this);
    mLabel->setStyleSheet(sQLabelStyle);
    //--------

    mShowOverlayCheckbox = new QCheckBox("Show overlay", this);
    mShowOverlayCheckbox->setStyleSheet(sQCheckBoxStyle);

    mTimeWindowWidth = 4; // in seconds, will show 5-1 = 4 seconds worth of plot
    float approxFPS = 15;
    mLastValues.set_capacity(mTimeWindowWidth * approxFPS);


    QPen penED(qRgb(255, 255, 100));
    penED.setStyle(Qt::SolidLine);
    penED.setWidth(3);

    QPen penES(qRgb(255, 150, 100));
    penES.setStyle(Qt::SolidLine);
    penES.setWidth(3);


    mAreaSeries = new QtCharts::QLineSeries();
    mAreaSeries->append(0, 0);
    mEDSeries = new QtCharts::QScatterSeries();
    mESSeries = new QtCharts::QScatterSeries();

    mChart = new QtCharts::QChart();
    mChart->legend()->hide();
    mChart->addSeries(mAreaSeries);
    mChart->addSeries(mEDSeries);
    mChart->addSeries(mESSeries);
    QtCharts::QValueAxis *t = new QtCharts::QValueAxis();
    t->setTickCount(mTimeWindowWidth+1);
    t->setRange(0, mTimeWindowWidth);
    t->setLabelFormat("%.0f");
    t->setTitleText("Time (s)");
    mChart->addAxis(t, Qt::AlignBottom);
    QtCharts::QValueAxis *axis_y = new QtCharts::QValueAxis();
    axis_y->setTickCount(4);
    double min_area = 5000, max_area = 40000;
    axis_y->setRange(min_area, max_area);
//    axis_y->setLabelFormat("%.0f");
    axis_y->hide();
    mChart->addAxis(axis_y, Qt::AlignLeft);

    mAreaSeries->attachAxis(t);
    mAreaSeries->attachAxis(axis_y);
    mEDSeries->attachAxis(t);
    mEDSeries->attachAxis(axis_y);
    mESSeries->attachAxis(t);
    mESSeries->attachAxis(axis_y);


    //    mChart->setTitle("LV volume [ml]");

    QtCharts::QChartView *chartView = new QtCharts::QChartView(mChart);
    //    chartView->layout()->setContentsMargins(0, 0, 0, 0);
    mChart->setBackgroundRoundness(0);
    mChart->setMargins(QMargins(0,0,0,0));
    mChart->setTheme(QtCharts::QChart::ChartThemeDark);
    QBrush blackBrush(QColor(0,0,0));
    mChart->setBackgroundBrush(blackBrush);
    mEDSeries->setPen(penED);
    mEDSeries->setMarkerSize(5);
    mESSeries->setPen(penES);
    mESSeries->setMarkerSize(3);

    //    const QString sQChartStyle = "QChart { border: 2px solid white; background-color : black; color : white; } ";
    //            "QPushButton::pressed {border: 2px solid white; background :  rgb(50, 150, 255); color: white;}"
    //            "QPushButton::checked {border: 2px solid white; background :  rgb(50, 150, 255); color: white;}";
    //    mChart->setStyleSheet(sQChartStyle);


    auto chartLayout = new QVBoxLayout(chartView);
    chartView->setMinimumHeight(150);
    //    chartView->setMinimumWidth(500);
    //    chartLayout->SetMinimumSize(QSize(300, 300));
    //    mChart->setPlotArea(QRectF(0,0,200,230));
    //    chartView->setRenderHint(QPainter::Antialiasing);



    /*
    mSlider = new QSlider(Qt::Orientation::Horizontal);
    mSlider->setStyleSheet(QtPluginWidgetBase::sQSliderStyle);

    mSlider->setMaximum(101);
    mSlider->setMinimum(0);
    mSlider->setAutoFillBackground(true);
    */

    mSaveButton = new QPushButton("Save", this);
    mSaveButton->setStyleSheet(sQPushButtonStyle);

    auto vLayout = new QVBoxLayout(this);
    vLayout->setContentsMargins(0, 0, 0, 0);
    vLayout->setSpacing(0);
    this->setLayout(vLayout);
    vLayout->addWidget(mLabel);
    //    vLayout->addLayout(chartLayout);
    vLayout->addWidget(chartView);
    this->AddInputStreamComboboxToLayout(vLayout);
    this->AddImageViewCheckboxToLayout(vLayout);
}

void Widget_LVSeg::Build(){

    //    auto labelFont = mLabel->font();
    //    labelFont.setPixelSize(15);
    //    labelFont.setBold(true);
    //    mLabel->setFont(labelFont);

    QVBoxLayout * outmost_layout = reinterpret_cast <QVBoxLayout *>( this->layout());
    //outmost_layout->addWidget(mLabel, 1, Qt::AlignTop);

    {
        QHBoxLayout *layout  = new QHBoxLayout();
        mShowOverlayCheckbox->setChecked(true);
        layout->addWidget(mShowOverlayCheckbox);
        layout->addWidget(mSaveButton);
        //layout->addStretch();
        outmost_layout->addLayout(layout);
    }


}

void Widget_LVSeg::SendImageToWidgetImpl(ifind::Image::Pointer image){

    if (mIsBuilt == false){
        mIsBuilt = true;
        this->Build();
    }

    std::stringstream stream;
    stream << "==" << this->mPluginName.toStdString() << "=="<<std::endl;
    //stream << "Receiving " << ifind::StreamTypeSetToString(this->mInputStreamTypes) << std::endl;
    if (image->HasKey("LVSeg_area")){
        //        stream << "Area: "<< image->GetMetaData<std::string>("LVSeg_area");
        //        stream << " [" << image->GetMetaData<std::string>("LVSeg_aream");
        //        stream << ", " <<  image->GetMetaData<std::string>("LVSeg_areaM") <<"]" <<std::endl;
        //        stream << "Area av: " <<  image->GetMetaData<std::string>("LVSeg_areaav")<<std::endl;

        float area_mm2 = QString(image->GetMetaData<std::string>("LVSeg_areamm2").c_str()).toFloat();
        float time_s = QString(image->GetMetaData<std::string>("LVSeg_timeseg").c_str()).toFloat();
        float hr = QString(image->GetMetaData<std::string>("LVSeg_HR").c_str()).toFloat();
        float ef = QString(image->GetMetaData<std::string>("LVSeg_EF").c_str()).toFloat();
        float EDV = QString(image->GetMetaData<std::string>("LVSeg_EDVol").c_str()).toFloat();
        float ESV = QString(image->GetMetaData<std::string>("LVSeg_ESVol").c_str()).toFloat();
        stream << "Area: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<  area_mm2/100.0 << " cm2"<<std::endl;
                stream << "HR: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<   hr << " bpm"<<std::endl;
                stream << "EF: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<   ef *100 << " %"<<std::endl;
                stream << "EDV: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<   EDV << " ml"<<std::endl;
                stream << "ESV: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<   ESV << " ml"<<std::endl;
                stream << "SV: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<   EDV-ESV << " ml"<<std::endl;
                stream << "CO: " << std::setw(3)<< std::setprecision(1)<< std::fixed<<   (EDV-ESV) * hr / 1000.0 << " l/min"<<std::endl;



        //        mSeries->append(mTime++, QRandomGenerator::global()->bounded(10));
        QtCharts::QValueAxis * axis_x = qobject_cast<QtCharts::QValueAxis *>(mChart->axes()[0]);
        float axmin = axis_x->min(), axmax = axis_x->max();
        float range = axmax-axmin;



        mLastValues.push_back(area_mm2/100.0);

        double current_t =  time_s;
//        mTime++;


        mAreaSeries->append(current_t, area_mm2/100.0);
//        std::cout << "(" << current_t <<", "<< area_mm2/100 << ")" <<std::endl;

        if (image->GetMetaData<std::string>("LVSeg_EDdetected").compare("True") == 0){
            int index = QString(image->GetMetaData<std::string>("LVSeg_ED").c_str()).toInt();
            QPointF point = mAreaSeries->at(mAreaSeries->count()+index);
            mEDSeries->append(point.x(), point.y()); //TODO change this
//            mEDSeries->append(current_t, area_mm2/100.0); //TODO change this
            if (mEDSeries->count() > mTimeWindowWidth){
                mEDSeries->remove(0);
            }
        }
        if (image->GetMetaData<std::string>("LVSeg_ESdetected").compare("True") == 0){
            int index = QString(image->GetMetaData<std::string>("LVSeg_ES").c_str()).toInt();
            QPointF point = mAreaSeries->at(mAreaSeries->count()+index);
            mESSeries->append(point.x(), point.y()); //TODO change this
//            mESSeries->append(current_t, area_mm2/100.0); //TODO change this
            if (mESSeries->count() > mTimeWindowWidth){
                mESSeries->remove(0);
            }
        }


        /// NOTE: potential fix here: https://stackoverflow.com/questions/46343422/qt-scrolling-the-visible-area-of-the-chart

        qreal At = 0;
        const int CHART_X_RANGE_COUNT = mLastValues.capacity();
        const int CHART_X_RANGE_MAX = CHART_X_RANGE_COUNT - 1;

        if (current_t > axmax-mTimeWindowWidth*0.1){
            At = mChart->plotArea().width() / CHART_X_RANGE_COUNT;

            QPointF p1 = mAreaSeries->at(mAreaSeries->count()-1);
            QPointF p0 = mAreaSeries->at(mAreaSeries->count()-2);
            float diff_s = p1.x()-p0.x();

            //float ticks_per_s = mChart->plotArea().width() / CHART_X_RANGE_COUNT  / mTimeWindowWidth;
            float ticks_per_s = mChart->plotArea().width() / mTimeWindowWidth;

//            std::cout << "Elapsed time: = "<< diff_s << "s real At  "<< mChart->plotArea().width() / CHART_X_RANGE_COUNT
//                      << ", area width= "<< mChart->plotArea().width() << ", nticks="<< CHART_X_RANGE_COUNT
//                      << ", ticks per second "<< ticks_per_s << ", window time ="<< mTimeWindowWidth << "s"
//                      << ", ticks for the difference = "<< diff_s * ticks_per_s<< std::endl;

            if (diff_s > 0.1){
                //std::cout << "there has been a long pause, extend the catch up from "<< At << " to "<< At +p1.x() - axis_x->min()<<std::endl;
                // if there is a wait of > 0.2 sec
                //At += (p1.x() - axis_x->min()) * CHART_X_RANGE_COUNT;

                At = diff_s * ticks_per_s;

            }
            mChart->scroll(At, 0);
            //            mESSeries->remove(0); // TODO change this
        }

        if (current_t < axmin){
            mAreaSeries->remove(0); // to not make it infinite
        }

        axmin = axis_x->min();
        axmax = axis_x->max();
        //        std::cout << "At= "<< At <<", axis ("<< axmin <<", "<< axmax<< ", current t: "<< current_t <<std::endl;

        QtCharts::QValueAxis * axis_y = qobject_cast<QtCharts::QValueAxis *>(mChart->axes()[1]);
        typedef boost::circular_buffer<double>::const_iterator iterator;
        std::pair<iterator, iterator> minMaxRawX =
                boost::minmax_element(mLastValues.begin(), mLastValues.end());

        axis_y->setRange(*minMaxRawX.first*0.95, *minMaxRawX.second*1.05);
        //mTimeWindowWidth

    }
    stream << "Sending " << ifind::StreamTypeSetToString(this->mStreamTypes);



    mLabel->setText(stream.str().c_str());
    Q_EMIT this->ImageAvailable(image);
}
