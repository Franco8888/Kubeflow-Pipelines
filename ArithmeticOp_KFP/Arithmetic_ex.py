import kfp
import kfp.components as comp
from kfp import compiler
import kfp.dsl as dsl
import sys


def check_args():
    '''Check if user provided a hostName, path and pipeline arguments
    Command line arugments are as follows:
    hostname 'pl-package-filename' pl-arg1 pl-arg2 pl-arg3 pl-arg4'''
    try:
        hostName = sys.argv[1]
        pl_arg1 = sys.argv[2]
        pl_arg2 = sys.argv[3]
        pl_arg3 = sys.argv[4]
        pl_arg4 = sys.argv[5]

    except:
        print()
        print("======Incorrect arguments provided=======")
        print("Command line arguments: hostname pl-arg1 pl-arg2 pl-arg3 pl-arg4")
        print()
        sys.exit()

    args_list = [hostName, pl_arg1, pl_arg2, pl_arg3, pl_arg4]
    return args_list


def get_Client(hostName):
    '''gets the client of the AI online platform using the hostname'''
    client = kfp.Client(host=hostName)
    return client


def create_pl_comp():
    ''''creates pipeline components using python functions. return list of components'''

    # Define your components code as standalone python functions:======================
    def add(a: float, b: float) -> float:
        '''Calculates sum of two arguments'''
        return a + b

    def multiply(c:float, d:float) ->float:
        '''Calculates the product'''
        return c*d

    # convert the python functions to a task factory (function that return a task object)
    add_op = comp.create_component_from_func(add, output_component_file='add_component.yaml', )
    # factory function used to create kfp.dsl.ContainerOp class instances for your pipeline

    add_op.component_spec.save('add_component.yaml')
    #add_op.component.OutputTextFile('Output.txt')

    # product_op is a task factory that creates a task object when given argument
    product_op = comp.create_component_from_func(multiply, output_component_file='multiple_component.yaml')

    component_lst = [add_op,product_op]

    return component_lst


def createPL(pl_comp_list):

    @dsl.pipeline(
      name='Addition-pipeline',
      description='An example pipeline that performs addition calculations.'
    )
    def my_pipeline(a, b='7', c='4', d='1'):

        add_op = pl_comp_list[0]
        product_op = pl_comp_list[1]

        # Uses task factory to create a task by passing arguments to the task factory.
        first_add_task = add_op(a, 4)
        # Passes an output reference from `first_add_task` and a pipeline parameter
        # to the `add_op` factory function. For operations with a single return
        # value, the output reference can be accessed as `task.output` or
        # `task.outputs['output_name']`.
        second_add_task = add_op(first_add_task.output, b)
        third_task = product_op(second_add_task.output, c)
        fourth_task = product_op(third_task.output, second_add_task.output)

    return my_pipeline

def runPipeline(pl, client, args1, args2, args3, args4):
    '''Run the pipeline on Ai platform pipeline and get output of the run'''

    # create arguments for PL
    arguments = {'a': args1, 'b': args2, 'c':args3, 'd':args4}  # arguments need to be in a dict

    exp_name = input("Provide Experiment Name: ")
    runName = input("Provide the run name: ")

    # this function runs a pipeline on KFP-enabled using client you created. returns run id
    runResult_Obj = client.create_run_from_pipeline_func(pl, arguments=arguments, run_name=runName,experiment_name=exp_name)

    runResult_Obj.wait_for_run_completion()


def package_pl(pl):
    '''Uploads a pipeline to the Ai platform Pipeline'''

    path = input("Please provide the path + filename for the pipeline package,\n"
                 "e.g. /home/user/PycharmProjects/SE-MLpipelines/iris_pl_package.yaml:\n")
    while(not path.endswith(".yaml")):
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
        print("======" + pl_name +" already exists on the AI Pipeline platform, Terminating program=======")
        print()
        sys.exit()
    print("Pipeline uploaded!")


def main():
    args_list = check_args()
    # args_list is as follows: hostname 'pathToSave-PipelinePackage' pl-arg1 pl-arg2 pl-arg3 pl-arg4
    hostName = args_list[0]

    client = get_Client(hostName)
    pl_comp_list = create_pl_comp()
    pl = createPL(pl_comp_list)
    runPipeline(pl, client, args_list[1], args_list[2], args_list[3], args_list[4])

    # Allows user to package PL and upload PL to AI platform. Can;t upload PL without packaging the PL
    print()
    packagePL = input("\nWould you like to package the pipeline to a .yaml file? Yes/No \n"
                      " Note: If you package the pipeline then you will be able to upload the PL to the AI platform PL:\n")
    if "yes" in packagePL.lower():
        path = package_pl(pl)

        upload = input("Would you like to upload the pipeline to the Ai platfrom Pipeline? Yes/No:\n")
        if "yes" in upload.lower():
            upload_pl(client, path)


if __name__ == "__main__":
    main()


