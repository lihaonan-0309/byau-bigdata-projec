<template>
    <div class="layout">
        <NavBarVue></NavBarVue>
        <el-container class="main-contain">
            <el-aside width="240px" class="aside">
                <SliderBarVue></SliderBarVue>
            </el-aside>
            <el-main class="main">
                <RouterView />
            </el-main>
        </el-container>
    </div>
</template>
<script lang="ts" setup>
import NavBarVue from './components/NavBar.vue';
import SliderBarVue from './components/SliderBar.vue';
</script>

<style lang="scss" scoped>
.layout{
    width: 100%;
    height: 100%;
}
.main-contain{
    height: calc(100% - 60px);
    &>.aside{
        height: 100%;
        background: #1E222D;
    }
    &>.main{
        height: 100%;
        background-color: rgba(116,116,128,0.08);
        &::-webkit-scrollbar {
            width: 8px;
        }

        &::-webkit-scrollbar-thumb {
            /*滚动条里面小方块*/
            box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.1);
            background: #cccccc;
            border-radius: 3px;
        }

        &::-webkit-scrollbar-track {
            /*滚动条里面轨道*/
            box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.1);
            background: #ededed;
        }
    }
}
</style>