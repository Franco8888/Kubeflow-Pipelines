import kfp
import kfp.compiler as compiler
import kfp.components as comp
import kfp.dsl as dsl
import sys


def connectToClient():
    '''Check to see if user entered host name and returns the client to the AI platform pipeline'''
    try:
        hostName = sys.argv[1]
    except:
        print()
        print("======No hostName provided. Terminating program=======")
        print("To get the host name go to your Kubeflow pipeline instance on the AI platform pipelines and click "
              "settings")
        print()
        sys.exit()

    client = kfp.Client(host=hostName)  # connect to KF pipeline
    return client


def create_pl_comp():
    '''create pipeline components. return list of components'''

    # INGEST function
    def ingest_task(output_data_path: comp.OutputPath(bytes), output_target_path: comp.OutputPath(bytes)):
        # import packages inside component function
        from sklearn.datasets import load_iris
        import numpy as np
        import joblib
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket('dev_kfp')

        iris = load_iris()
        data = iris.data  # array of the data
        target = iris.target  # array of labels (i.e answers) of each data entry

        # store output to binary files and upload to GCS
        with open(output_data_path, 'wb') as fo:  # doctest: +ELLIPSIS
            joblib.dump(data, fo)

        # read the same file and upload to GCS
        with open(output_data_path, 'rb') as fo:
            blob = bucket.blob('iris-ingest-data-final')
            blob.upload_from_file(fo)

        with open(output_target_path, 'wb') as fo:  # doctest: +ELLIPSIS
            joblib.dump(target, fo)

        # read the same file and upload to GCS
        with open(output_target_path, 'rb') as fo:
            blob = bucket.blob('iris-ingest-target-final')
            blob.upload_from_file(fo)

    # DATA PREP function
    # taking random indices to split the dataset into train and test
    def data_prep_task(
            data_path: comp.InputPath(),
            target_path: comp.InputPath(),
            output_dataTrain_path: comp.OutputPath(bytes),
            output_dataTest_path: comp.OutputPath(bytes),
            output_targetTrain_path: comp.OutputPath(bytes),
            output_targetTest_path: comp.OutputPath(bytes),
    ):

        # import packages inside component function
        import numpy as np
        import joblib
        from google.cloud import storage
        from sklearn.model_selection import train_test_split

        storage_client = storage.Client()
        bucket = storage_client.bucket('dev_kfp')

        # read from input files
        with open(data_path, 'rb') as fo:
            data = joblib.load(fo)
        with open(target_path, 'rb') as fo:
            target = joblib.load(fo)


        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2,
                                                                            random_state=42)

        # store output to binary files
        with open(output_dataTrain_path, 'wb') as fo:
            joblib.dump(data_train, fo)
        with open(output_dataTrain_path, 'rb') as fo:
            blob = bucket.blob('iris-prep-training-data-final')
            blob.upload_from_file(fo)

        with open(output_dataTest_path, 'wb') as fo:
            joblib.dump(data_test, fo)
        with open(output_dataTest_path, 'rb') as fo:
            blob = bucket.blob('iris-prep-testing-data-final')
            blob.upload_from_file(fo)

        with open(output_targetTrain_path, 'wb') as fo:
            joblib.dump(target_train, fo)
        with open(output_targetTrain_path, 'rb') as fo:
            blob = bucket.blob('iris-prep-training-target-final')
            blob.upload_from_file(fo)

        with open(output_targetTest_path, 'wb') as fo:
            joblib.dump(target_test, fo)
        with open(output_targetTest_path, 'rb') as fo:
            blob = bucket.blob('iris-prep-testing-target-final')
            blob.upload_from_file(fo)

    def training_task(
            dataTrain_path: comp.InputPath(),
            targetTrain_path: comp.InputPath(),
            output_clf_path: comp.OutputPath(bytes)
    ):

        from sklearn import tree
        import joblib
        import numpy as np
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket('dev_kfp')

        # read from input files
        with open(dataTrain_path, 'rb') as fo:
            data_train = joblib.load(fo)
        with open(targetTrain_path, 'rb') as fo:
            target_train = joblib.load(fo)

        # train data
        clf = tree.DecisionTreeClassifier()
        clf.fit(data_train, target_train)

        # store output to binary files
        with open(output_clf_path, 'wb') as fo:
            joblib.dump(clf, fo)
        with open(output_clf_path, 'rb') as fo:
            blob = bucket.blob('iris-training-classifier-final')
            blob.upload_from_file(fo)

    # PREDICTION function
    # predictions on the test dataset
    def prediction_task(
            clf_path: comp.InputPath(),
            dataTest_path: comp.InputPath(),
            targetTest_path: comp.InputPath(), ):

        from sklearn.metrics import accuracy_score
        import numpy as np
        import joblib
        from google.cloud import storage
        from io import BytesIO

        storage_client = storage.Client()
        bucket = storage_client.bucket('dev_kfp')

        # read from input files
        with open(clf_path, 'rb') as fo:
            clf = joblib.load(fo)
        with open(dataTest_path, 'rb') as fo:
            data_test = joblib.load(fo)
        with open(targetTest_path, 'rb') as fo:
            target_test = joblib.load(fo)

        # predict data
        pred = clf.predict(data_test)
        print("Prediction:     ", pred)  # predicted labels i.e flower species
        print("Target_test:    ", target_test)  # actual labels
        print("Accuracy Score: ", (accuracy_score(pred, target_test)) * 100)  # prediction accuracy

        # persist prediction to bucket
        with open('temp_prediction_path', 'wb') as fo:
            joblib.dump(pred, fo)
        with open('temp_prediction_path', 'rb') as fo:
            blob = bucket.blob('iris-prediction-prediction-final')
            blob.upload_from_file(fo)


    # Allows user to package PL components to yaml files
    packagePLc = input("\nWould you like to package the pipeline components to yaml files? Yes/No \n")
    if "yes" in packagePLc.lower():
        # create pl components
        ingest_comp = comp.func_to_container_op(ingest_task,
                                                packages_to_install=['sklearn', 'google-cloud-storage', 'numpy==1.19.2',
                                                                     'scikit-learn==0.23.2',
                                                                     'joblib==0.17.0'],
                                                output_component_file='iris_ingest_comp.yaml')
        data_prep_comp = comp.func_to_container_op(data_prep_task,
                                                   packages_to_install=['sklearn', 'google-cloud-storage', 'numpy==1.19.2',
                                                                        'joblib==0.17.0'],
                                                   output_component_file='iris_dataPrep_comp.yaml')
        training_comp = comp.func_to_container_op(training_task,
                                                  packages_to_install=['google-cloud-storage', 'numpy==1.19.2',
                                                                       'scikit-learn==0.23.2', 'joblib==0.17.0'],
                                                  output_component_file='iris_training_comp.yaml')
        prediction_comp = comp.func_to_container_op(prediction_task,
                                                    packages_to_install=['google-cloud-storage', 'numpy==1.19.2',
                                                                         'scikit-learn==0.23.2', 'joblib==0.17.0'],
                                                    output_component_file='iris_prediction_comp.yaml')

    else:
        # create pl components
        ingest_comp = comp.func_to_container_op(ingest_task,
                                                packages_to_install=['sklearn', 'google-cloud-storage', 'numpy==1.19.2',
                                                                     'scikit-learn==0.23.2', 'joblib==0.17.0'])
        data_prep_comp = comp.func_to_container_op(data_prep_task,
                                                   packages_to_install=['sklearn', 'google-cloud-storage', 'numpy==1.19.2',
                                                                        'joblib==0.17.0'])
        training_comp = comp.func_to_container_op(training_task,
                                                  packages_to_install=['google-cloud-storage', 'numpy==1.19.2',
                                                                       'scikit-learn==0.23.2', 'joblib==0.17.0'])
        prediction_comp = comp.func_to_container_op(prediction_task,
                                                    packages_to_install=['google-cloud-storage', 'numpy==1.19.2',
                                                                         'scikit-learn==0.23.2', 'joblib==0.17.0'])

    comp_list = [ingest_comp, data_prep_comp, training_comp, prediction_comp]

    return comp_list


def create_pl(pl_comp_list):
    """creates a pipeline using the pipeline components. Returns a pipeline"""

    @dsl.pipeline(
        name='Iris-Pipeline',
        description='A pipeline which implements the iris data set'
    )
    def my_pipeline():
        ingest_comp = pl_comp_list[0]
        data_prep_comp = pl_comp_list[1]
        training_comp = pl_comp_list[2]
        prediction_comp = pl_comp_list[3]

        # implement pipeline tasks using components
        ingest_task = ingest_comp()
        data_prep_task = data_prep_comp(ingest_task.outputs['output_data'], ingest_task.outputs['output_target'])
        training_task = training_comp(data_prep_task.outputs['output_dataTrain'],
                                      data_prep_task.outputs['output_targetTrain'])
        prediction_task = prediction_comp(training_task.outputs['output_clf'],
                                          data_prep_task.outputs['output_dataTest'],
                                          data_prep_task.outputs['output_targetTest'])

    return my_pipeline


def run_pipeline(client, pl):
    exp_name = input("Provide Experiment Name: ")
    runName = input("Provide the run name: ")

    timeout = input("Please give the timeout(seconds) for the running of the pipeline, i.e. If the PL is not completed"
                    " within this time the PL will stop processing. Make sure you provide enough time ")

    print()
    print("Running Pipeline, please wait")


    # this function runs a pipeline on KFP-enabled using client you created. returns run id
    runObj = client.create_run_from_pipeline_func(pl, arguments=dict(), run_name=runName,
                                                         experiment_name=exp_name)

    try:
        runObj.wait_for_run_completion(float(timeout))  # increase this amount if you expect your pipeline to run for a long time
    except:
        print()
        print("======PL could not finish wihtin specified time, increase timeout=======")
        print()
        sys.exit()


    print("Pipeline finished running! ")

def package_pl(pl):
    """Uploads a pipeline to the Ai platform Pipeline"""

    path = input("Please provide the path + filename for the pipeline package,\n"
                 "e.g. /home/user/PycharmProjects/SE-MLpipelines/iris_pl_package.yaml:\n")
    while (not path.endswith(".yaml")):
        path = input("Incorrect format of file, please re-enter the file name,\n"
                     "e.g. /home/user/PycharmProjects/SE-MLpipelines/iris_pl_package.yaml:\n")
    # compile pipeline into a package
    compiler.Compiler().compile(pl, path)  # compile pipeline into package

    return path


def upload_pl(client, pl_package_path):
    pl_name = input("\nProvide the name of the Pipeline which will be saved on AI platform pipeline: ")
    pl_description = input("Provide Pipeline description: ")

    # upload the pipeline package to the AI platform pipeline
    try:
        # Note: This will throw an error if the pipeline already exists
        client.upload_pipeline(pipeline_package_path=pl_package_path,
                               pipeline_name=pl_name, description=pl_description)
    except:
        print()
        print("======" + pl_name + " already exists on the AI Pipeline platform, Terminating program=======")
        print()
        sys.exit()
    print("Pipeline uploaded!")


def print_prediction_accuracy():
    import joblib
    from google.cloud import storage
    from io import BytesIO
    from sklearn.metrics import accuracy_score

    # access bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket('dev_kfp')

    # get test target values from GCS
    gcs_test_target = bucket.blob('iris-prep-testing-target-final')
    buffer_test_target = BytesIO()
    gcs_test_target.download_to_file(buffer_test_target)
    buffer_test_target.seek(0)
    test_target_gcs = joblib.load(buffer_test_target)

    # get prediction object from GCS
    gcs_prediction = bucket.blob('iris-prediction-prediction-final')
    buffer_pred = BytesIO()
    gcs_prediction.download_to_file(buffer_pred)
    buffer_pred.seek(0)
    prediction_object_gcs = joblib.load(buffer_pred)

    print("Prediction:     ", prediction_object_gcs)  # predicted labels i.e flower species
    print("Target_test:    ", test_target_gcs)  # actual labels
    print("Accuracy Score: ", (accuracy_score(prediction_object_gcs, test_target_gcs)) * 100)  # prediction accuracy


def entry_point():
    client = connectToClient()
    pl_comp_list = create_pl_comp()
    pl = create_pl(pl_comp_list)
    run_pipeline(client, pl)

    # Allows user to package and upload PL to AI platform. Can;t upload PL without packaging the PL
    packagePL = input("\nWould you like to package the pipeline to a .yaml file? Yes/No \n"
                      "Note: If you package the pipeline then you will be able to upload the PL to the AI platform PL: ")
    if "yes" in packagePL.lower():
        path = package_pl(pl)

        upload = input("Would you like to upload the pipeline to the Ai platfrom Pipeline? Yes/No: ")
        if "yes" in upload.lower():
            upload_pl(client, path)

    print_prediction_accuracy()


if __name__ == "__main__":
    entry_point()
