#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='oai_agents',
      version='0.1.0',
      description='Cooperative multi-agent environment based on Overcooked',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Stéphane Aroca-Ouellette',
      author_email='stephane.aroca-ouellette@colorado.edu',
      url='https://github.com/StephAO/oai_agents',
      download_url='https://github.com/StephAO/oai_agents',
      keywords=['Overcooked', 'AI', 'Reinforcement Learning', 'Human Agent Collaboration'],
      # packages=find_packages('oai_agents'),
      # package_dir={"": "oai_agents"},
      packages=['oai_agents', 'oai_agents.agents', 'oai_agents.gym_environments', 'oai_agents.common'],
      package_dir={
          'oai_agents': 'oai_agents',
          'oai_agents.agents': 'oai_agents/agents',
          'oai_agents.gym_environments': 'oai_agents/gym_environments',
          'oai_agents.common': 'oai_agents/common'
      },
      package_data={
        'oai_agents' : [
          'data/*.pickle'
        ],
      },
      install_requires=[
        'numpy',
        'stable-baselines3==1.7.0',
        'sb3-contrib==1.7.0',
        'gymnasium==0.28.1',
        'pylsl==1.16.2',
        'pickle5==0.0.12',
        'cloudpickle==2.2.1',
        # 'liblsl==1.16.2',
        'tqdm',
        'wandb',
        'pygame',
        'dill',
      ],
      tests_require=['pytest']
    )
