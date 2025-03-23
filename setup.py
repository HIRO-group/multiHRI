#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='oai_agents',
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
    )
