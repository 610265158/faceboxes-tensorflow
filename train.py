from base_trainer.net_work import trainner

import setproctitle
setproctitle.setproctitle("faceboxes")

trainner=trainner()

trainner.train()
