Hostfiles
=========

Hostfiles (sometimes called "machine files") are a combination of two
things:

#. A listing of hosts on which to launch processes.
#. Optionally, limit the number of processes which can be launched on
   each host.

Syntax
------

Hostfile syntax consists of one node name on each line, optionally
including a designated number of "slots":

.. code:: sh

   # This is a comment line, and will be ignored
   node01  slots=10
   node13  slots=5

   node15
   node16
   node17  slots=3
   ...

Blank lines and lines beginning with a ``#`` are ignored.

A "slot" is the PRRTE term for an allocatable unit where we can launch
a process.  :ref:`See this section
<placement-definition-of-slot-label>` for a longer description of
slots.

In the absence of the ``slot`` parameter, PRRTE will assign either the
number of slots to be the number of CPUs detected on the node or the
resource manager-assigned value if operating in the presence of an
RM.

.. important:: If using a resource manager, the user-specified number
               of slots is capped by the RM-assigned value.
