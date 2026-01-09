;;; SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
;;;                         All rights reserved.
;;; SPDX-License-Identifier: Apache-2.0
;;; legate specific configuration for Emacs
((nil . ((indent-tabs-mode . nil)
         (tab-width        . 4)
         (eval             . (add-hook 'before-save-hook #'delete-trailing-whitespace))))
 (python-mode . ((python-interpreter   . (seq-find (lambda (item) (executable-find item)) '("python3" "python")))
                 (python-indent-offset . 4)
                 (fill-column          . 79))))
