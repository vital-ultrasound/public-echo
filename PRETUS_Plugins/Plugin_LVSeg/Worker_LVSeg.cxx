#include "Worker_LVSeg.h"
#include <iostream>
#include <QDebug>
#include <list>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <itkImportImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkBinaryThresholdImageFilter.h>
#include "pngutils.hxx"
//#include "volumeutils.hxx"

Worker_LVSeg::Worker_LVSeg(QObject *parent) : Worker(parent){
    this->python_folder = "";
    this->modelname = "model.pth";
    this->params.out_size[0] = 128;
    this->params.out_size[1] = 128;
    this->params.out_spacing[0] = 9.0;
    this->params.out_spacing[1] = 6.0;
    this->params.origin = Worker::WorkerParameters::OriginPolicy::Centre;
    // TODO: I should probably do something about the spacing

    this->mAreaBuffer.clear();
    this->mTimeBuffer.clear();
    mEDTimeBuffer.clear();
    mESTimeBuffer.clear();
    mEDVolBuffer.clear();
    mESVolBuffer.clear();
    mLastDetectedED = 1; // this will be neg values actually
    mLastDetectedES = 1;
    mInitialTime = std::chrono::steady_clock::now();;
    mNCycles = 4;
    this->mMaxAreaBufferSize = 25 * mNCycles; // about 4 cycles worth of data
    this->mSegmentationThreshold = 128;

    mAbsoluteCropBounds = false; // by default relative
    mAratio = {275, 175};
    //mCropBounds = {500, 200, 1100, 700}; // for 1920 × 1080
    mCropBounds = {0.25, 0.2, 0.55, 0.65}; // for 1920 × 1080
    mDesiredSize = {this->params.out_size[0], this->params.out_size[1]};

    QObject::connect(this, &Worker_LVSeg::signal_PhaseDetected,
                     this, &Worker_LVSeg::slot_computeVentricularVolume);
}

void Worker_LVSeg::Initialize(){

    if (this->params.verbose){
        std::cout << "Worker_LVSeg::Initialize()"<<std::endl;
    }

    if (!this->PythonInitialized){
        try {
            py::initialize_interpreter();
        }
        catch (py::error_already_set const &pythonErr) {
            std::cout << "[ERROR] Worker_LVSeg::Initialize() " << pythonErr.what();
        }

    }

    if (this->params.verbose){
        std::cout << "Worker_LVSeg::Initialize() - load model ..." << std::flush;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    {
        py::exec("import sys");
        std::string command = "sys.path.append('" + this->python_folder + "')";
        py::exec(command.c_str());

        py::object processing;
        try {
            processing = py::module::import("LVSeg_worker");
        } catch (pybind11::error_already_set & err) {
            std::cout << "[ERROR] Worker_LVSeg::Initialize() - error while importing LVSeg module" << std::endl;
            err.restore();
        }
        /// Check for errors
        if (PyErr_Occurred())
        {
            std::cout << "[ERROR] Worker_LVSeg::Initialize() " << std::endl;
            PyErr_Print();
            return;
        }

        /// grabbing the functions from module
        this->PyImageProcessingFunction = processing.attr("dowork");
        this->PyPythonInitializeFunction = processing.attr("initialize");
        this->PyVolumeComputationFunction = processing.attr("segmentation_to_volume");
        py::tuple sz = py::make_tuple(mDesiredSize[0], mDesiredSize[1]);
        this->PyPythonInitializeFunction(sz,this->python_folder + "/model", this->modelname, this->mSegmentationThreshold);

        this->PythonInitialized = true;
    }
    PyGILState_Release(gstate);

    if (this->params.verbose){
        std::cout << "loaded"<<std::endl;
    }

}

Worker_LVSeg::~Worker_LVSeg(){
    /// Finalize python stuff
    py::finalize_interpreter();
}

void Worker_LVSeg::doWork(ifind::Image::Pointer image){


    if (!this->PythonInitialized){
        return;
    }

    if (!Worker::gil_init) {
        Worker::gil_init = 1;
        PyEval_InitThreads();
        PyEval_SaveThread();

        ifind::Image::Pointer configuration = ifind::Image::New();
        configuration->SetMetaData<std::string>("Python_gil_init","True");
        Q_EMIT this->ConfigurationGenerated(configuration);
    }

    if (image == nullptr){
        if (this->params.verbose){
            std::cout << "Worker_LVSeg::doWork() - input image was null" <<std::endl;
        }
        return;
    }

    if (image->HasKey("IsPaused")){
        if (QString(image->GetMetaData<std::string>("IsPaused").c_str()).toInt() == 1){
            return;
        }
    }



    if (this->params.verbose){
        std::cout << "Worker_LVSeg::doWork()"<<std::endl;
    }

    /// Extract central slice and crop
    if (this->params.verbose){
        std::cout << "Worker_LVSeg::doWork() - adjust ratio"<<std::endl;
    }


    ifind::Image::Pointer image_ratio_adjusted;
    // get the image size
    ifind::Image::SizeType imsize = image->GetLargestPossibleRegion().GetSize();
    std::vector<int> absoluteCropBounds(4);
    if (this->absoluteCropBounds() == true){
        std::copy(mCropBounds.begin(), mCropBounds.end(), back_inserter(absoluteCropBounds));
    } else {

        absoluteCropBounds[0] = int(mCropBounds[0] * imsize[0]); // x0
        absoluteCropBounds[1] = int(mCropBounds[1] * imsize[1]); // y0
        absoluteCropBounds[2] = int(mCropBounds[2] * imsize[0]); // w
        absoluteCropBounds[3] = int(mCropBounds[3] * imsize[1]); // h

        if (this->params.verbose){
            std::cout << "\tWorker_LVSeg::doWork() computing absolute crop bounds"<<std::endl;
            std::cout << "\t\timage size is "<< imsize[0] << "x" << imsize[1]<<std::endl;
            std::cout << "\t\trelative crop bounds are "<< mCropBounds[0] << ":" << mCropBounds[1]<< ":" << mCropBounds[2]<< ":" << mCropBounds[3]<<std::endl;
            std::cout << "\t\tabsolute crop bounds are "<< absoluteCropBounds[0] << ":" << absoluteCropBounds[1]<< ":" << absoluteCropBounds[2]<< ":" << absoluteCropBounds[3]<<std::endl;
        }
    }

    /// Use the appropriate layer
    std::vector<std::string> layernames = image->GetLayerNames();
    int layer_idx = this->params.inputLayer;
    if (this->params.inputLayer <0){
        /// counting from the end
        layer_idx = image->GetNumberOfLayers() + this->params.inputLayer;
    }
    ifind::Image::Pointer layerImage = ifind::Image::New();
    layerImage->Graft(image->GetOverlay(layer_idx), layernames[layer_idx]);
    image_ratio_adjusted = this->CropImageToFixedAspectRatio(layerImage, &mAratio[0], &absoluteCropBounds[0]);

    //png::save_ifind_to_png_file<ifind::Image>(image_ratio_adjusted, "/home/ag09/data/VITAL/cpp_in_adjusted.png");
    // now resample to 128 128
    if (this->params.verbose){
        std::cout << "Worker_LVSeg::doWork() - resample"<<std::endl;
    }
    ifind::Image::Pointer image_ratio_adjusted_resampled  = this->ResampleToFixedSize(image_ratio_adjusted, &mDesiredSize[0]);
    //png::save_ifind_to_png_file<ifind::Image>(image_ratio_adjusted_resampled, "/home/ag09/data/VITAL/cpp_in_adjusted_resampled.png");
    this->params.out_spacing[0] = this->params.out_spacing[0] * (this->params.out_size[0] - 1 )/ (128 - 1);
    this->params.out_spacing[1] = this->params.out_spacing[1] * (this->params.out_size[1] - 1 )/ (128 - 1);
    this->params.out_size[0] = 128;
    this->params.out_size[1] = 128;


    GrayImageType2D::Pointer image_2d = this->get2dimage(image_ratio_adjusted_resampled);

    GrayImageType2D::Pointer lv_segmentation;

    /// Create a numpy array containing the image scalars
    /// Input dimensions are swapped as ITK and numpy have inverted orders
    std::vector <unsigned long> dims = {image_2d->GetLargestPossibleRegion().GetSize()[1], image_2d->GetLargestPossibleRegion().GetSize()[0]};
    if (!image_2d->GetBufferPointer() || (dims[0] < 50) || (dims[1] < 50))
    {
        qWarning() << "[Worker_LVSeg] image buffer is invalid";
        return;
    }

    double area_pixels = 0;
    double area_mm = 0;
    double area_mm2 = 0;
    this->gstate = PyGILState_Ensure();
    {

        py::array numpyarray(dims, static_cast<GrayImageType::PixelType*>(image_2d->GetBufferPointer()));
        py::tuple im_spacing = py::make_tuple(image_2d->GetSpacing()[0], image_2d->GetSpacing()[1]);


//        this->PyPythonInitializeFunction(sz,this->python_folder + "/model", this->modelname);

        py::object _function = this->PyImageProcessingFunction;
        /// predict biometrics
        py::tuple out_tuple = py::array(_function(numpyarray, im_spacing));
        py::array segmentation_array = out_tuple[0];
        py::float_ vol_ml_array = out_tuple[1];
        area_mm2 = vol_ml_array;
        //area_pixels = out_tuple[1].cast<int>();

        /// ---------- Get the segmentation of the fitted ellipse -----------------------

        typedef itk::ImportImageFilter< GrayImageType::PixelType, 2 >   ImportFilterType;
        ImportFilterType::SizeType imagesize;

        imagesize[0] = segmentation_array.shape(1);
        imagesize[1] = segmentation_array.shape(0);

        if (this->params.verbose){
            std::cout << "Worker_LVSeg::doWork() - image size is "<< imagesize[0]<< " x " << imagesize[1] <<std::endl;
        }


        ImportFilterType::RegionType region;
        ImportFilterType::IndexType start;
        start.Fill(0);

        region.SetIndex(start);
        region.SetSize(imagesize);

        /// Define import filter
        ImportFilterType::Pointer importer = ImportFilterType::New();
        importer->SetOrigin( image_2d->GetOrigin() );
        importer->SetSpacing( image_2d->GetSpacing() );
        importer->SetDirection( image_2d->GetDirection() );
        importer->SetRegion(region);
        /// Separate the regional scalar buffer
        /// @todo check if a memcpy is necessary here
        GrayImageType::PixelType* localbuffer = static_cast<GrayImageType::PixelType*>(segmentation_array.mutable_data());
        /// Import the buffer
        importer->SetImportPointer(localbuffer, imagesize[0] * imagesize[1], false);
        importer->Update();

        /// Disconnect the output from the filter
        /// @todo Check if that is sufficient to release the numpy buffer, or if the buffer needs to obe memcpy'ed
        lv_segmentation = importer->GetOutput();
        lv_segmentation->DisconnectPipeline();

        lv_segmentation->SetMetaDataDictionary(image_2d->GetMetaDataDictionary());

        /// ---------- Get the contour of the ellipse --------- ---------------------------
        /// Create a 3D image with the 2D slice
        GrayImageType::Pointer segmentation = get3dimagefrom2d(lv_segmentation);
        //png::save_ifind_to_png_file<GrayImageType>(segmentation, "/home/ag09/data/VITAL/segmentation.png");
        GrayImageType::Pointer segmentation_unresized= this->UndoResampleToFixedSize(segmentation, image, &absoluteCropBounds[0]);
        GrayImageType::Pointer responsemap;

        if (this->mSegmentationThreshold >=0){
            GrayImageType::PixelType threshold = this->mSegmentationThreshold;
            using FilterType = itk::BinaryThresholdImageFilter<GrayImageType, GrayImageType>;
            auto filter = FilterType::New();
            filter->SetInput(this->UndoCropImageToFixedAspectRatio(segmentation_unresized, image, &absoluteCropBounds[0]));
            filter->SetLowerThreshold(threshold);
            //filter->SetUpperThreshold(UpperThreshold);
            filter->SetOutsideValue(0);
            filter->SetInsideValue(255);
            filter->Update();
            responsemap = filter->GetOutput();
        } else {
            responsemap = this->UndoCropImageToFixedAspectRatio(segmentation_unresized, image, &absoluteCropBounds[0]);
        }
        //GrayImageType::Pointer responsemap = this->UnAdjustImageSize(segmentation, image);
        //png::save_ifind_to_png_file<GrayImageType>(segmentation_unresized, "/home/ag09/data/VITAL/unresampled_seg.png");
        //GrayImageType::Pointer responsemap = this->UndoCropImageToFixedAspectRatio(segmentation_unresized, image, &absoluteCropBounds[0]);

        image->GraftOverlay(responsemap.GetPointer(), image->GetNumberOfLayers(), "Segmentation");
        image->SetMetaData<std::string>( mPluginName.toStdString() +"_output", QString::number(image->GetNumberOfLayers()).toStdString() );


        {
            area_pixels = 0;

            using GrayIteratorType = itk::ImageRegionConstIterator<GrayImageType>;
            GrayIteratorType it( responsemap, responsemap->GetRequestedRegion() );

            for ( it.GoToBegin() ; !it.IsAtEnd(); ++it)
            {
                area_pixels += int(it.Value()>150) ; // only foreground pixels
            }
            area_mm= area_pixels * image->GetSpacing()[0] * image->GetSpacing()[1];
        }
        //png::save_ifind_to_png_file<ifind::Image>(image, "/home/ag09/data/VITAL/input_image.png");
        //png::save_ifind_to_png_file<GrayImageType>(responsemap, "/home/ag09/data/VITAL/segmentaiton_image.png");

        if (this->params.verbose){
            std::cout << "\tWorker_LVSeg::doWork() - done" <<std::endl;
        }
    }
    PyGILState_Release(this->gstate);



//    float timestamp_s = atof(image->GetMetaData<std::string>("DNLTimestamp").c_str())/1000.0; // This is not working, but I do not know why
    auto now = std::chrono::steady_clock::now();
    float timestamp_s =  std::chrono::duration_cast<std::chrono::milliseconds>(now - mInitialTime).count()/1000.0; //std::chrono::duration<double> diff = end - start;

    // deal with the area buffer - ensure the size is appropriate
    this->mAreaBuffer.enqueue(area_mm2);


    this->mTimeBuffer.enqueue(timestamp_s);
    while (this->mAreaBuffer.size()>this->mMaxAreaBufferSize+1){
        this->mAreaBuffer.dequeue();
        this->mTimeBuffer.dequeue();
    }
    // compute min, max and average
    int M = -1;
    int m = 10000000;
    float area_average = 0;
    for (auto l : this->mAreaBuffer){
        if (l > M) M = l;
        if (l < m) m = l;
        area_average += l;
    }
    area_average /= this->mAreaBuffer.size();


    image->SetMetaData<std::string>( mPluginName.toStdString() +"_area", QString::number(area_mm).toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_areaM", QString::number(M).toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_aream", QString::number(m).toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_areaav", QString::number(area_average).toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_areamm2", QString::number(area_mm2).toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_timeseg", QString::number(timestamp_s).toStdString() );
    // send the indices of the ED frames relative to the current one (the current one would be 0)
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_ED", "0" );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_ES", "0" );


    int ED_index, ES_index;
    bool detected_ED, detected_ES;
    bool exclusive = true;

    this->detect_ED_ES(ED_index, ES_index, detected_ED, detected_ES, exclusive);


    if (detected_ED == true){
        // trigger EDV calculation using the simpsons rule on a separate thread
        Q_EMIT this->signal_PhaseDetected(image, area_mm2/100, true);

        image->SetMetaData<std::string>( mPluginName.toStdString() +"_EDdetected", "True" );
        image->SetMetaData<std::string>( mPluginName.toStdString() +"_ED", QString::number(ED_index).toStdString() ); // here send the index where the peak was detected


        float ed_time = this->mTimeBuffer[this->mTimeBuffer.count()+ED_index];
        mEDTimeBuffer.enqueue(ed_time);

    } else {
        image->SetMetaData<std::string>( mPluginName.toStdString() +"_EDdetected", "False" );
    }

    if (detected_ES == true){
        Q_EMIT this->signal_PhaseDetected(image, area_mm2/100.0, false);

        image->SetMetaData<std::string>( mPluginName.toStdString() +"_ESdetected", "True" );
        image->SetMetaData<std::string>( mPluginName.toStdString() +"_ES", QString::number(ES_index).toStdString() ); // here send the index where the peak was detected

        float es_time = this->mTimeBuffer[this->mTimeBuffer.count()+ES_index];
        mESTimeBuffer.enqueue(es_time);

    } else {
        image->SetMetaData<std::string>( mPluginName.toStdString() +"_ESdetected", "False" );
    }


    float heart_rate = this->estimate_hr();
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_HR", QString::number(heart_rate).toStdString() );

    float ejection_fraction = this->estimate_ef();
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_EF", QString::number(ejection_fraction).toStdString() );


    image->SetMetaData<std::string>( mPluginName.toStdString() +"_EDVol", QString::number(mEDVolBuffer[mEDVolBuffer.count()-1]).toStdString() );
    image->SetMetaData<std::string>( mPluginName.toStdString() +"_ESVol", QString::number(mESVolBuffer[mESVolBuffer.count()-1]).toStdString() );


    Q_EMIT this->ImageProcessed(image);


    if (this->params.verbose){
        std::cout << "[VERBOSE] Worker_LVSeg::doWork() - image processed." <<std::endl;
    }

    //exit(-1);

}

void Worker_LVSeg::slot_computeVentricularVolume(ifind::Image::Pointer image, float area_cm2, bool isED){


    // do the avove in python
    int nlayers = image->GetNumberOfLayers();
    std::vector<std::string> layernames = image->GetLayerNames();
    ifind::Image::Pointer segmentation = ifind::Image::New();
    segmentation->Graft(image->GetOverlay(nlayers-1), layernames[nlayers-1]);
    GrayImageType2D::Pointer image_2d = this->get2dimage(segmentation);
    std::vector <unsigned long> dims = {image_2d->GetLargestPossibleRegion().GetSize()[1], image_2d->GetLargestPossibleRegion().GetSize()[0]};
    float vol_ml;
    this->gstate = PyGILState_Ensure();
    {
        py::array numpyarray(dims, static_cast<GrayImageType::PixelType*>(image_2d->GetBufferPointer()));
        py::tuple im_spacing = py::make_tuple(image_2d->GetSpacing()[0], image_2d->GetSpacing()[1]);

        py::object _function = this->PyVolumeComputationFunction;
        py::float_ value = py::array(_function(numpyarray, im_spacing));
        vol_ml = value;

    }
    PyGILState_Release(this->gstate);

//    // here implement the simpsons rule
//    vol_ml = area_cm2;

    if (isED == true){

        mEDVolBuffer.enqueue(vol_ml);
        if (mEDVolBuffer.count() > mNCycles){
            mEDVolBuffer.dequeue();
        }
    } else {
//        std::cout << "Worker_LVSeg::slot_computeVentricularVolume:  ES volume = "<< vol_ml << "ml" <<std::endl;
        mESVolBuffer.enqueue(vol_ml);
        if (mEDVolBuffer.count() > mNCycles){
            mESVolBuffer.dequeue();
        }
    }
}

float Worker_LVSeg::estimate_ef(){

    int n_cycles = mNCycles;
    unsigned int n_outliers_at_each_end = 1;
    /// Compute using just the last values:
    n_cycles = 1;
    n_outliers_at_each_end = 0;


    std::list<float> efs;

    for (int i=1; i < n_cycles+1; i++){
        if ( (mEDVolBuffer.count() > i) && (mESVolBuffer.count() > i) ){
            float SV = mEDVolBuffer[mEDVolBuffer.count()-i] - mESVolBuffer[mESVolBuffer.count()-i];
            float ef = SV/mEDVolBuffer[mEDVolBuffer.count()-i];
            efs.push_back(ef);
        }
    }


    if (efs.size() <= n_outliers_at_each_end * 2){
        return 0;
    }


    // sort to remove outliers
    efs.sort();

    float avg = 0;
    std::list<float>::iterator it = efs.begin();
    std::list<float>::iterator end = efs.end();
    for (int j=0; j < n_outliers_at_each_end; j++){
        ++it;
        --end;
    }
    int n_times = 0;
    for(; it != end; it++){
        avg += *it;
        n_times++;
    }
    avg /= n_times;

    return avg;
}

float Worker_LVSeg::estimate_hr(){

    int n_cycles = mNCycles;
    unsigned int n_outliers_at_each_end = 1;
    /// Compute using just the last values:
    n_cycles = 1;
    n_outliers_at_each_end = 0;


    std::list<float> cycle_times;

    for (int i=1; i < n_cycles+1; i++){
        if (mEDTimeBuffer.count() > i){
            float cycle_time = mEDTimeBuffer[mEDTimeBuffer.count()-i] - mEDTimeBuffer[mEDTimeBuffer.count()-(i+1)];
            cycle_times.push_back(cycle_time);
        }
        if (mESTimeBuffer.count() > i){
            float cycle_time = mESTimeBuffer[mESTimeBuffer.count()-i] - mESTimeBuffer[mESTimeBuffer.count()-(i+1)];
            cycle_times.push_back(cycle_time);
        }
    }



    if (cycle_times.size() <= n_outliers_at_each_end * 2){
        return 0;
    }


    // sort to remove outliers
    cycle_times.sort();

    float avg = 0;
    std::list<float>::iterator it = cycle_times.begin();
    std::list<float>::iterator end = cycle_times.end();
    for (int j=0; j < n_outliers_at_each_end; j++){
        ++it;
        --end;
    }
    int n_times = 0;
    for(; it != end; it++){
        avg += *it;
        n_times++;
    }
    avg /= n_times;

    return 60.0 / avg;

}

void Worker_LVSeg::detect_ED_ES(int &ED_index, int &ES_index, bool &detected_ED, bool &detected_ES, bool exclusive){

//    std::cout << "Worker_LVSeg::detect_ED_ES"<<std::endl;

//    for (int i =0 ; i < this->mAreaBuffer.count(); i++){
//        std::cout << this->mAreaBuffer[i] << ", ";
//    }
//    std::cout << std::endl;

//    for (int i =0 ; i < this->mTimeBuffer.count(); i++){
//        std::cout << this->mTimeBuffer[i] << ", ";
//    }
//    std::cout << std::endl;


    ED_index = 0;
    ES_index = 0;
    detected_ED = false;
    detected_ES = false;

    int minimum_inter_peak_samples = 5;

    // updatethe position of the last detected frames

    if (mLastDetectedED < 0){
        mLastDetectedED--;
        if (-mLastDetectedED >= this->mAreaBuffer.count() ){
            mLastDetectedED = 1;
        }
    }

    if (mLastDetectedES < 0){
        mLastDetectedES--;
        if (-mLastDetectedES >= this->mAreaBuffer.count() ){
            mLastDetectedES = 1;
        }
    }

    int interval_lenght = 30;
    float max = -1, min=1e10;
    int max_id = 0, min_id = 0;
    for (int i = this->mAreaBuffer.count()-1; i > this->mAreaBuffer.count()-interval_lenght-1; --i){
        if ((this->mAreaBuffer[i] > max) &&
                (this->mAreaBuffer[i-1] < this->mAreaBuffer[i]) &&
                (this->mAreaBuffer[i+1] < this->mAreaBuffer[i])){
            max = this->mAreaBuffer[i];
            max_id = i - this->mAreaBuffer.count();
        }
    }
    for (int i = this->mAreaBuffer.count()-1; i > this->mAreaBuffer.count()-interval_lenght-1; --i){
        if ((this->mAreaBuffer[i] < min) &&
                (this->mAreaBuffer[i-1] > this->mAreaBuffer[i]) &&
                (this->mAreaBuffer[i+1] > this->mAreaBuffer[i])){
            min= this->mAreaBuffer[i];
            min_id = i - this->mAreaBuffer.count();
        }
    }


    if (max_id < 0){
//        std::cout << "Max: "<< max <<", max index = "<< max_id<< " last: "<< mLastDetectedED<< std::endl;
        if ((max_id > mLastDetectedED + minimum_inter_peak_samples) || (mLastDetectedED >= 0)){
            ED_index = max_id; //+1-this->mAreaBuffer.count();
//            std::cout << "  Detected ED is true "<< ED_index<< " ("<< mLastDetectedED<< ") "<< (ED_index > mLastDetectedED) << std::endl;
            mLastDetectedED = ED_index;
            detected_ED = true;
        }
    }

    if (min_id < 0){
//        std::cout << "Min: "<< min <<", min index = "<< min_id<< " last: "<< mLastDetectedED<< std::endl;
        if ((min_id > mLastDetectedES + minimum_inter_peak_samples) || (mLastDetectedES >= 0)){
            ES_index = min_id; //+1-this->mAreaBuffer.count();
//            std::cout << "  Detected ES is true "<< ES_index<< " ("<< mLastDetectedES<< ") "<< (ES_index > mLastDetectedES) << std::endl;
            mLastDetectedES = ES_index;
            detected_ES = true;
        }
    }

    if (exclusive){
        if (detected_ED && detected_ES ){
            // make true only the most recent one
            if (ED_index > ES_index){
                detected_ES = false;
            } else {
                detected_ED = false;
            }
        }
    }

}

bool Worker_LVSeg::absoluteCropBounds() const
{
    return mAbsoluteCropBounds;
}

void Worker_LVSeg::setAbsoluteCropBounds(bool absoluteCropBounds)
{
    mAbsoluteCropBounds = absoluteCropBounds;
}

std::vector<int> Worker_LVSeg::desiredSize() const
{
    return mDesiredSize;
}

void Worker_LVSeg::setDesiredSize(const std::vector<int> &desiredSize)
{
    mDesiredSize = desiredSize;
}

std::vector<float> Worker_LVSeg::aratio() const
{
    return mAratio;
}

void Worker_LVSeg::setAratio(const std::vector<float> &aratio)
{
    mAratio = aratio;
}

std::vector<double> Worker_LVSeg::cropBounds() const
{
    return mCropBounds;
}

void Worker_LVSeg::setCropBounds(const std::vector<double> &cropBounds)
{
    mCropBounds = cropBounds;
}
