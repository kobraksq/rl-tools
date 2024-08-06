#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DEVICES_CPU_OPENBLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DEVICES_CPU_OPENBLAS_H

#include "../utils/generic/typing.h"
#include "devices.h"

#include "cpu_blas.h"

#include <cblas.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices{
    template <typename T_SPEC>
    struct CPU_OPENBLAS: CPU_BLAS<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_OPENBLAS;
    };
    using DefaultCPU_OPENBLAS = CPU_OPENBLAS<DefaultCPUSpecification>;
}
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC>
    void init(devices::CPU_OPENBLAS<DEV_SPEC>& device){
//        openblas_set_num_threads(4);
        if (!device.initialized) {
            time_t now;
            time(&now);
            char buf[sizeof "0000-00-00T00:00:00Z"];
            strftime(buf, sizeof buf, "%FT%TZ", localtime(&now));
            device.run_name = devices::cpu::sanitize_file_name(buf);
            device.runs_path = std::string("runs");
            device.run_path = device.runs_path + "/" + device.run_name;
            device.initialized = true;
        }
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
