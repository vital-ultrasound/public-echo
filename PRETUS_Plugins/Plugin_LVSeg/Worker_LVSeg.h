#pragma once

#include <Worker.h>
#include <QQueue>
#include <memory>
#include <vector>
#include <chrono>

/// For image data. Change if image data is different
#include <ifindImage.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

class Worker_LVSeg  : public Worker{
    Q_OBJECT

public:

    typedef Worker_LVSeg            Self;
    typedef std::shared_ptr<Self>       Pointer;

    /** Constructor */
    static Pointer New(QObject *parent = 0) {
        return Pointer(new Self(parent));
    }

    ~Worker_LVSeg();

    void Initialize();

    /// parameters must be only in the parent class
    std::string python_folder;
    std::string modelname;

    std::vector<double> cropBounds() const;
    void setCropBounds(const std::vector<double> &cropBounds);

    std::vector<float> aratio() const;
    void setAratio(const std::vector<float> &aratio);

    std::vector<int> desiredSize() const;
    void setDesiredSize(const std::vector<int> &desiredSize);

    bool absoluteCropBounds() const;
    void setAbsoluteCropBounds(bool absoluteCropBounds);

    int mSegmentationThreshold;


public Q_SLOTS:
    virtual void slot_computeVentricularVolume(ifind::Image::Pointer image, float area_cm2, bool isED);

Q_SIGNALS:
    void signal_PhaseDetected(ifind::Image::Pointer image, float area_cm2, bool isED);


protected:
    Worker_LVSeg(QObject* parent = 0);

    void doWork(ifind::Image::Pointer image);

    void detect_ED_ES(int &ED_index, int &ES_index, bool &detected_ED, bool &detected_ES, bool exclusive=false);
    float estimate_hr();
    float estimate_ef();

    QQueue<float> mAreaBuffer;
    QQueue<float> mTimeBuffer;
    QQueue<float> mEDTimeBuffer;
    QQueue<float> mESTimeBuffer;
    QQueue<float> mEDVolBuffer;
    QQueue<float> mESVolBuffer;
    int mLastDetectedED;
    int mLastDetectedES;
    std::chrono::time_point<std::chrono::steady_clock> mInitialTime;
    int mMaxAreaBufferSize;
    int mNCycles;

    /**
     * @brief mCropBounds
     * x0. y0. width, height
     */
    std::vector<double> mCropBounds;
    std::vector<float> mAratio;
    std::vector<int> mDesiredSize;
    bool mAbsoluteCropBounds;


private:

    /// Python Functions
    py::object PyImageProcessingFunction;
    py::object PyVolumeComputationFunction;
    py::object PyPythonInitializeFunction;
    PyGILState_STATE gstate;

};
