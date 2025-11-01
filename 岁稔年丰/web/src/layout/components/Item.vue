<template>
    <div v-if="!props.item.hidden">
        <template v-if="hasOneShowingChild(props.item.children,props.item) && (!onlyOneChild.children || onlyOneChild.noShowingChildren)">
            <router-link v-if="onlyOneChild.meta" :to="resolvePath(onlyOneChild.path)">
                <el-menu-item :index="resolvePath(onlyOneChild.path)">
                    <img width="16" v-if="onlyOneChild.meta.icon" :src="$route.fullPath == onlyOneChild.path || $route.fullPath == props.item.path ? onlyOneChild.meta.activeIcon || (props.item.meta && props.item.meta.activeIcon) : onlyOneChild.meta.icon || (props.item.meta && props.item.meta.icon)" style="margin-right: 20px;" alt="" />
                    <span>{{onlyOneChild.meta.title}}</span>
                </el-menu-item>
            </router-link>
        </template>
        <el-sub-menu v-else ref="submenu" :index="resolvePath(props.item.path)">
            <template #title>
                <img width="16" v-if="props.item.meta.icon" :src="$route.fullPath.indexOf(props.item.path) != -1 && props.item.meta ? props.item.meta.activeIcon : props.item.meta.icon" alt="" style="margin-right: 20px;" />
                <span>{{props.item.meta.title}}</span>
            </template>
            <Item v-for="child in item.children" :key="child.path" :item="child" :base-path="resolvePath(child.path)"></Item>
        </el-sub-menu>
    </div>
</template>

<script lang="ts" setup>
import { watch,ref } from 'vue';
import { useRouter } from 'vue-router';
import { isExternal } from '@/utils/validate';
import path from 'path-browserify';

const props = defineProps(['item','basePath']);
let router = useRouter();

let onlyOneChild:any = null;
const hasOneShowingChild = (children = [],parent:{}) => {
    const showingChildren = children.filter((item) => {
        if(item['hidden'] === true){
            return false;
        }else{
            onlyOneChild = item;
            return true;
        }
    });
    if(showingChildren.length === 1 && onlyOneChild.meta.icon){
        return true;
    }
    if(showingChildren.length === 0){
        onlyOneChild = {...parent,path:"",noShowingChildren:true};
        return true;
    }
    return false;
}
const resolvePath = (routePath:string):string => {
  if(isExternal(routePath)){
        return routePath;
    }
    if (isExternal(props.basePath)) {
        return props.basePath;
    }
    return path.resolve(props.basePath, routePath);
}
</script>

<style scoped>
    a{
        text-decoration: none !important;
    }
</style>