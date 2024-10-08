import logging
from copy import copy
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Literal, Optional, cast, Any, Union, TypedDict
from collections import OrderedDict

import networkx as nx
import numpy as np
import yaml

logger = logging.getLogger(__name__)  # ! this is not functional I think
TASK_M_DEF = 'nomad.datamodel.metainfo.workflow.TaskReference'
WORKFLOW_M_DEF = 'nomad.datamodel.metainfo.workflow.Workflow'

SectionType = Literal['task', 'workflow', 'input', 'output']


# Define a custom representer for OrderedDict
def represent_ordereddict(dumper, data):
    return dumper.represent_dict(data.items())


# Register the custom representer
yaml.add_representer(OrderedDict, represent_ordereddict)


class ArchivePathInfo(TypedDict, total=False):
    step_index: int = Field(None, description='Snapshot number')
    run_number: int = Field(None, description='Run number')
    upload_id: Optional[str] = Field(None, description='Upload ID')
    entry_id: Optional[str] = Field(None, description='Entry ID')
    mainfile_path: Optional[str] = Field(
        None, description='Main file'
    )  # ! This is not optional!
    path_type: Optional[str] = Field(None, description='subsection type') # TODO this needs a better name
    full_path: str = Field('', description='Archive path')


class NomadSection(BaseModel):  # ! This is where I do the path logic
    name: Optional[str] = Field(None, description='Name of the section')
    type: Optional[SectionType] = Field(None, description='Type of the section')
    # label: Optional[str] = Field(None, description='Label of the section')
    archive_path_info: ArchivePathInfo = Field(
        default_factory=dict, description='Archive path'
    )
    inputs: List[Dict[str, Any]] = Field(
        [{}],
        description='section inputs',
    )
    outputs: List[Dict[str, Any]] = Field([{}], description='section outputs')

    def __init__(self, **data):
        super().__init__(**data)

        if self.archive_path_info:
            if self.archive_path_info.get('full_path'):
                self.archive_path = self.archive_path_info['full_path']
            else:
                if self.archive_path_info.get('run_number') is None:
                    self.archive_path_info.run_number = -1
                if self.step_index is None:  # ! check these
                    self.step_index = (
                        -1 if self.type == 'output' else 0
                    )  # ? Something like this
                if self.archive_path is None:
                    self.archive_path = f'run/{self.run_number}/calculation/{self.step_index}'
        else:
            self.archive_path = ''

    @property
    def upload_suffix(self) -> str: # HERE I AM!!


    @property
    def upload_prefix(self) -> str:
        if self.entry_id:
            upload_prefix = f'/uploads/{self.entry_id}'
        elif self.upload_id:
            upload_prefix = f'/uploads/{self.upload_id}'
        else:
            upload_prefix = '../upload'

        return (
            f'{upload_prefix}/archive/mainfile/{self.mainfile_path}'
            if self.mainfile_path
            else ''
        )

    @property
    def full_path(self) -> str:
        return f'{self.upload_prefix}#/{self.archive_path}'

    def to_dict(self) -> dict:
        return OrderedDict({'name': self.name, 'section': self.full_path})


class NomadTask(BaseModel):
    name: str
    m_def: str
    inputs: List[NomadSection] = Field(default_factory=list)
    outputs: List[NomadSection] = Field(default_factory=list)
    task_section: Optional[NomadSection] = None

    # class Config:
    #     arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        for i, input in enumerate(self.inputs):
            if input.name is None:
                input.name = f'input_{i}'
        for o, output in enumerate(self.outputs):
            if output.name is None:
                output.name = f'output_{o}'

    @property
    def m_def(self) -> str:
        if self.task_section.type == 'workflow':
            return WORKFLOW_M_DEF
        elif self.task_section.type == 'task':
            return TASK_M_DEF

    @property
    def task(self) -> Optional[str]:
        if self.task_section.type == 'workflow':
            return (
                self.task_section.upload_prefix + '#/workflow2'
            )  # TODO probably need to check if full path is given somehow
        else:
            return None

    def to_dict(self) -> dict:
        output_dict = OrderedDict()
        if self.m_def:
            output_dict['m_def'] = self.m_def
        output_dict['name'] = self.name
        if self.task:
            output_dict['task'] = self.task
        output_dict['inputs'] = [i.to_dict() for i in self.inputs]
        output_dict['outputs'] = [o.to_dict() for o in self.outputs]

        return output_dict


class NomadWorkflowArchive(BaseModel):
    name: str = 'workflow2'
    inputs: list[NomadSection] = Field(default_factory=list)
    outputs: list[NomadSection] = Field(default_factory=list)
    tasks: list[NomadTask] = Field(default_factory=list)

    # class Config:
    #     arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        return {
            self.name: OrderedDict(
                {
                    'inputs': [i.to_dict() for i in self.inputs],
                    'outputs': [o.to_dict() for o in self.outputs],
                    'tasks': [t.to_dict() for t in self.tasks],
                }
            ),
        }

    def to_yaml(self, destination_filename: str) -> None:
        with open(destination_filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    # @classmethod
    # def from_multiple_jobs(
    #     cls, project: MartiniFlowProject, jobs: list[Job], aggregate_same_task_names: bool = True
    # ) -> "NomadWorkflowArchive":
    #     def filter_unique(ele):
    #         final_inputs = []
    #         for inp in ele:
    #             if inp not in final_inputs:
    #                 final_inputs.append(inp)
    #         return final_inputs

    #     archive = NomadWorkflowArchive()
    #     for job in jobs:
    #         workflow = NomadWorkflow(project, job, is_top_level=True)
    #         job_inputs = [inp.add_job_id(job) for inp in copy(workflow.generate_archive().inputs)]
    #         archive.inputs.extend(job_inputs)
    #         job_outputs = [out.add_job_id(job) for out in copy(workflow.generate_archive().outputs)]
    #         archive.outputs.extend(job_outputs)
    #         job_tasks = copy(workflow.generate_archive().tasks)

    #         for task in job_tasks:
    #             task.inputs = [inp.add_job_id(job) for inp in task.inputs]
    #             task.outputs = [out.add_job_id(job) for out in task.outputs]
    #             task.task_section = task.task_section.add_job_id(job)
    #         archive.tasks.extend(job_tasks)

    #     if aggregate_same_task_names:
    #         final_tasks = []
    #         for task in archive.tasks:
    #             if task.name not in [t.name for t in final_tasks]:
    #                 final_tasks.append(task)
    #             else:
    #                 dest_task = next(t for t in final_tasks if t.name == task.name)
    #                 dest_task.inputs = filter_unique(dest_task.inputs)
    #                 dest_task.outputs = filter_unique(dest_task.outputs)
    #         archive.tasks = final_tasks

    #     archive.inputs = filter_unique(archive.inputs)
    #     archive.outputs = filter_unique(archive.outputs)
    #     archive.tasks = filter_unique(archive.tasks)
    #     return archive


class NomadWorkflow(BaseModel):
    destination_filename: str
    node_attributes: Dict[int, Any] = {}
    task_elements: Dict[str, NomadSection] = Field(default_factory=dict)
    # task_counter: int = 0

    # class Config:
    #     arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.task_elements = {}
        # self.task_counter = 0

        # @property
        # def project_name(self) -> str:
        #     return self.project.__class__.__name__

        # @property
        # def gromacs_logs(self) -> dict:
        #     return self.job.doc[self.project_name].get("gromacs_logs", {})

        # @property
        # def tasks(self) -> dict:
        #     return self.job.doc[self.project_name].get('tasks', {})

        # @property
        # def workflows(self) -> dict:
        #     return self.job.doc[self.project_name].get("workflows", {}) if self.is_top_level else {}

        # @property
        # def all_tasks(self) -> dict:
        #     return dict(self.gromacs_logs) | dict(self.tasks) | dict(self.workflows)

    def register_section(
        self, node_key: Union[int, str, tuple], node_attrs: Dict[str, Any]
    ) -> None:
        # section_type = self._section_type(operation_name)  # ! coming from input dict
        # label = self.all_tasks[operation_name]  # ! coming from input dict
        # if section_type == 'workflow':
        #     label = self.job.doc[
        #         self.all_tasks[operation_name]
        #     ].get(
        #         'nomad_workflow', self.all_tasks[operation_name]
        #     )  # ! this is labeling with pre-defined yaml name, but not sure where it is coming from...in any case, it would be if we define standard workflows within
        # upload_id = (
        #     self.job.doc[self.all_tasks[operation_name]].get('nomad_upload_id', None)
        #     if section_type == 'workflow'
        #     else None
        # )  # ! don't we want to use entry_ids for this?
        section = NomadSection(**node_attrs)
        # if section.is_task:
        #     self.task_counter += 1
        # if self.add_job_id:
        #     section.add_job_id(self.job)
        print(node_key, section)
        self.task_elements[node_key] = section  # ! build the tasks section by section
        print(self.task_elements[node_key])

        # @cached_property
        # def graph(self) -> nx.DiGraph:
        #     operations = list(self.project.operations.keys())
        #     adjacency_matrix = np.asarray(self.project.detect_operation_graph())
        #     signac_graph = nx.DiGraph(adjacency_matrix)
        #     graph = nx.DiGraph()
        #     all_tasks = dict(self.gromacs_logs) | dict(self.tasks) | dict(self.workflows)
        #     for node_index in signac_graph.nodes:
        #         op_name = operations[node_index]
        #         if op_name in all_tasks:
        #             graph.add_node(op_name, label=all_tasks[op_name], is_task=op_name in self.tasks)
        #             self.register_section(op_name)
        #     for node_1, node_2 in signac_graph.edges:
        #         if (op_name_1 := operations[node_1]) in graph.nodes and (op_name_2 := operations[node_2]) in graph.nodes:
        #             graph.add_edge(op_name_1, op_name_2)
        #     return graph
        # @cached_property
        # def graph(self) -> nx.DiGraph:
        #     operations = self.node_names
        #     adjacency_matrix = self.adjaceny_matrix
        #     signac_graph = nx.DiGraph(adjacency_matrix)
        #     graph = nx.DiGraph()
        #     for node_index in signac_graph.nodes:
        #         op_name = operations[node_index]
        #         graph.add_node(
        #             op_name,
        #             label=self.node_types[node_index],
        #             is_task=self.node_types[node_index] == 'task',
        #         )
        #         self.register_section(op_name)
        #     for node_1, node_2 in signac_graph.edges:
        #         if (op_name_1 := operations[node_1]) in graph.nodes and (
        #             op_name_2 := operations[node_2]
        #         ) in graph.nodes:
        #             graph.add_edge(op_name_1, op_name_2)
        #     return graph  # ! I guess the point of this new graph is to simply add the attributes?

    @property
    def workflow_graph(self) -> nx.DiGraph:
        # TODO treat case where someone makes their own graphs with some checks
        workflow_graph = nx.DiGraph()

        if self.node_attributes:
            workflow_graph.add_nodes_from(self.node_attributes.keys())
            nx.set_node_attributes(workflow_graph, self.node_attributes)

        for node_key, node_attrs in self.node_attributes.items():
            inputs = node_attrs.get('inputs', [])
            for input_ in inputs:
                edge_node = input_.get('node_reference')
                if edge_node is not None:
                    workflow_graph.add_edge(edge_node, node_key)
                else:  # add to the workflow inputs
                    node_index = len(workflow_graph.nodes)
                    workflow_graph.add_node(node_index, type='input', **input_)
                    workflow_graph.add_edge(node_index, node_key)

            outputs = node_attrs.get('outputs', [])
            for output_ in outputs:
                edge_node = output_.get('node_reference')
                if edge_node is not None:
                    workflow_graph.add_edge(node_key, edge_node)
                else:
                    node_index = len(workflow_graph.nodes)
                    workflow_graph.add_node(node_index, type='output', **output_)
                    workflow_graph.add_edge(node_key, node_index)

        return workflow_graph

    # def build_workflow_yaml(self, destination_filename: str) -> None:
    #     archive = self.generate_archive()
    #     archive.to_yaml(destination_filename)
    #     project_name = self.project.class_name()
    #     self.job.doc = update_nested_dict(self.job.doc, {project_name: {"nomad_workflow": destination_filename}})
    def build_workflow_yaml(self) -> None:
        # register the sections and build task_elements
        # register the nodes as sections for the archive construction
        for node_key, node_attrs in self.workflow_graph.nodes(data=True):
            self.register_section(node_key, node_attrs)

        archive = self.generate_archive()
        archive.to_yaml(self.destination_filename)
        # project_name = self.project.class_name()
        # self.job.doc = update_nested_dict(self.job.doc, {project_name: {"nomad_workflow": destination_filename}})

    def generate_archive(self) -> NomadWorkflowArchive:
        archive = NomadWorkflowArchive()
        archive.inputs = []
        archive.outputs = []

        for node_key, node in self.workflow_graph.nodes(data=True):
            if node.get('type', '') == 'input':
                element = self.task_elements[node_key]
                archive.inputs.append(element)
            elif node.get('type', '') == 'output':
                element = self.task_elements[node_key]
                element.step_index = -1  # ! check this
                archive.outputs.append(element)
            elif node.get('type', '') in ['task', 'workflow']:
                archive.tasks.append(
                    NomadTask(
                        name=node.get('name', ''),
                        inputs=node.get('inputs', []),
                        outputs=node.get('outputs', []),
                        task_section=self.task_elements[node_key],
                    )
                )
        return archive

    # def _section_type(self, operation_name: str) -> SectionType:
    #     # TODO Here instead I could make this a property and store the types upon init?
    #     if operation_name in self.tasks:
    #         return 'task'
    #     elif operation_name in self.workflows:
    #         return 'workflow'
    #     return 'default'


# @MartiniFlowProject.label
# def nomad_workflow_is_built(job: Job) -> bool:
#     project = cast(MartiniFlowProject, job.project)
#     return job.isfile(project.nomad_workflow)


# def build_nomad_workflow(job, is_top_level: bool = False, add_job_id: bool = False):
#     project = cast(MartiniFlowProject, job.project)
#     workflow = NomadWorkflow(project, job, is_top_level, add_job_id=add_job_id)
#     destination_filename = project.nomad_top_level_workflow if is_top_level else project.nomad_workflow
#     workflow.build_workflow_yaml(destination_filename)
def build_nomad_workflow(
    destination_filename: str = './nomad_workflow.archive.yaml',
    node_attributes: Dict[int, Any] = {},
    write_to_yaml: bool = False,
) -> nx.DiGraph:
    workflow = NomadWorkflow(
        destination_filename=destination_filename,
        node_attributes=node_attributes,
    )
    if write_to_yaml:
        workflow.build_workflow_yaml()

    return workflow.workflow_graph
